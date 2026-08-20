"""Measure Qwen generation, client filtering, and real Windows loopback onset.

This is intentionally an end-to-end diagnostic. It consumes each HTTP response,
plays the filtered PCM, and records the first non-silent frame from the default
WASAPI loopback device. Run it only on a quiet test machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from array import array
from pathlib import Path
from typing import Any, Callable


OUTPUT_RATE = 48_000
SOURCE_RATE = 24_000


class LoopbackOnset:
    def __init__(self, *, threshold: int = 300) -> None:
        import pyaudiowpatch as pyaudio

        self._pyaudio_module = pyaudio
        self._audio = pyaudio.PyAudio()
        self._device = self._audio.get_default_wasapi_loopback()
        self._channels = max(1, int(self._device["maxInputChannels"]))
        self._rate = int(round(float(self._device["defaultSampleRate"])))
        self._threshold = max(1, int(threshold))
        self._lock = threading.Lock()
        self._armed_at: float | None = None
        self._onset: float | None = None
        self._event = threading.Event()
        self._perf_anchor = 0.0
        self._pa_anchor = 0.0
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=int(self._device["index"]),
            frames_per_buffer=128,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        time.sleep(0.15)
        self._perf_anchor = time.perf_counter()
        self._pa_anchor = float(self._stream.get_time())

    @property
    def device_name(self) -> str:
        return str(self._device["name"])

    def arm(self, started: float) -> None:
        with self._lock:
            self._armed_at = float(started)
            self._onset = None
            self._event.clear()

    def wait(self, timeout: float) -> float | None:
        if not self._event.wait(timeout):
            return None
        with self._lock:
            return self._onset

    def _callback(
        self,
        data: bytes,
        frame_count: int,
        time_info: dict[str, float],
        _status: int,
    ) -> tuple[None, int]:
        with self._lock:
            armed_at = self._armed_at
            already_found = self._onset is not None
        if armed_at is not None and not already_found and data:
            samples = array("h")
            samples.frombytes(data)
            adc_time = float(time_info.get("input_buffer_adc_time", 0.0))
            for frame in range(min(frame_count, len(samples) // self._channels)):
                offset = frame * self._channels
                amplitude = max(
                    abs(int(samples[offset + channel]))
                    for channel in range(self._channels)
                )
                if amplitude < self._threshold:
                    continue
                sample_pa_time = adc_time + frame / float(self._rate)
                sample_perf_time = (
                    self._perf_anchor + sample_pa_time - self._pa_anchor
                )
                if sample_perf_time < armed_at:
                    continue
                with self._lock:
                    if self._onset is None:
                        self._onset = sample_perf_time
                        self._event.set()
                break
        return None, self._pyaudio_module.paContinue

    def close(self) -> None:
        try:
            self._stream.stop_stream()
        finally:
            self._stream.close()
            self._audio.terminate()


def _render_48k(filtered: bytes, previous: int | None) -> tuple[bytes, int]:
    import numpy as np

    samples = np.frombuffer(filtered, dtype="<i2").astype(np.int32)
    if not len(samples):
        return b"", previous or 0
    if previous is None:
        rendered = np.empty(max(1, len(samples) * 2 - 1), dtype=np.int32)
        rendered[0::2] = samples
        if len(samples) > 1:
            rendered[1::2] = (samples[:-1] + samples[1:]) // 2
    else:
        rendered = np.empty(max(1, len(samples) * 2), dtype=np.int32)
        rendered[0] = (previous + int(samples[0])) // 2
        rendered[1::2] = samples
        if len(samples) > 1:
            rendered[2:-1:2] = (samples[:-1] + samples[1:]) // 2
    gain = 1.15
    peak = int(np.max(np.abs(rendered)))
    if peak > 0:
        gain = min(gain, (32767.0 * 0.94) / peak)
    output = np.clip(rendered * gain, -32768, 32767).astype("<i2").tobytes()
    return output, int(samples[-1])


def _last_native_ttfa_ms(ssh: list[str]) -> float | None:
    command = (
        "grep '\\[Perf\\] TTFA ' "
        "/home/lon/Dev/wwz-commentator-runtime/qwen-tts.log | tail -n 1"
    )
    result = subprocess.run(
        [*ssh, command],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    marker = "[Perf] TTFA "
    line = result.stdout.strip()
    if marker not in line:
        return None
    return float(line.split(marker, 1)[1].split(" ms", 1)[0])


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return {}
    p95_index = max(0, min(len(values) - 1, int(len(values) * 0.95) - 1))
    return {
        "median": statistics.median(values),
        "p95": values[p95_index],
        "minimum": values[0],
        "maximum": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18086")
    parser.add_argument("--commentator", type=Path, required=True)
    parser.add_argument("--player", choices=("ffplay", "persistent"), required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--voice", default="mira_v5_spark")
    parser.add_argument("--threshold", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--ssh-host", default="lon@192.168.178.22")
    args = parser.parse_args()

    sys.path.insert(0, str(args.commentator.resolve()))
    from ai_gameplay_commentator.speech.pcm_output import (
        PcmPlayer,
        _FfplayProcessPcmPlayer,
        warm_pcm_output,
    )
    from ai_gameplay_commentator.speech.tts import _InitialSilenceFilter

    player_factory: Callable[[], Any]
    player_factory = (
        _FfplayProcessPcmPlayer if args.player == "ffplay" else PcmPlayer
    )
    if args.player == "persistent" and not warm_pcm_output():
        raise RuntimeError("isolated PCM sidecar failed to warm up")
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if args.ssh_key is not None:
        ssh.extend(("-i", str(args.ssh_key)))
    ssh.append(args.ssh_host)

    payload = json.dumps(
        {
            "input": "That was a close one, but we made it through.",
            "voice": args.voice,
            "response_format": "pcm",
            "seed": 42,
            "max_new_tokens": 128,
            "temperature": 0.9,
        }
    ).encode()
    loopback = LoopbackOnset(threshold=args.threshold)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(args.runs):
            silence_filter = _InitialSilenceFilter(sample_rate=SOURCE_RATE)
            previous_sample: int | None = None
            player = None
            request_started = time.perf_counter()
            loopback.arm(request_started)
            request = urllib.request.Request(
                args.endpoint.rstrip("/") + "/v1/audio/speech",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urllib.request.urlopen(request, timeout=30)
            header_received = time.perf_counter()
            first_network: float | None = None
            filter_released: float | None = None
            audio_seconds = 0.0
            while True:
                chunk = response.read1(8192)
                if not chunk:
                    break
                if first_network is None:
                    first_network = time.perf_counter()
                filtered = silence_filter.process(chunk)
                if not filtered:
                    continue
                if filter_released is None:
                    filter_released = time.perf_counter()
                    player = player_factory()
                rendered, previous_sample = _render_48k(filtered, previous_sample)
                player.append(rendered)
                audio_seconds += len(rendered) / 2.0 / OUTPUT_RATE
            response.close()
            if player is None:
                raise RuntimeError("server produced no PCM after the silence filter")
            first_callback = getattr(player, "playback_started_perf_counter", None)
            if first_callback is None:
                first_callback = getattr(player, "playback_started_monotonic", None)
            player.finish()
            loopback_onset = loopback.wait(2.0)
            player.wait(max(2.0, audio_seconds + 1.0))
            player.close(abort=False)
            native_ttfa_ms = _last_native_ttfa_ms(ssh)
            row = {
                "index": index,
                "player": args.player,
                "native_first_frame_ms": native_ttfa_ms,
                "http_header_ms": (header_received - request_started) * 1000,
                "first_network_ms": (
                    (first_network - request_started) * 1000
                    if first_network is not None
                    else None
                ),
                "silence_filter_release_ms": (
                    (filter_released - request_started) * 1000
                    if filter_released is not None
                    else None
                ),
                "player_callback_ms": (
                    (float(first_callback) - request_started) * 1000
                    if first_callback is not None
                    else None
                ),
                "loopback_audible_ms": (
                    (loopback_onset - request_started) * 1000
                    if loopback_onset is not None
                    else None
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            time.sleep(0.2)
    finally:
        loopback.close()

    keys = (
        "native_first_frame_ms",
        "http_header_ms",
        "first_network_ms",
        "silence_filter_release_ms",
        "player_callback_ms",
        "loopback_audible_ms",
    )
    report = {
        "player": args.player,
        "runs": rows,
        "summary_ms": {key: _summary(rows, key) for key in keys},
        "loopback_device": loopback.device_name,
        "threshold": args.threshold,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
