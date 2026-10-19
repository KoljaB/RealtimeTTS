"""Own a packaged Breeze worker for the lifetime of the compatible API."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import socket
import subprocess
import time

import requests


@contextmanager
def local_worker(model: Path, voice_dir: Path, *, timeout: float = 180,
                 chunk_first: int = 4, chunk_max: int = 6):
    if not 1 <= chunk_first <= chunk_max <= 64:
        raise ValueError("Breeze chunks must satisfy 1 <= first <= max <= 64")
    model = model.expanduser().absolute()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("--model must name a Breeze TTS 2 Q8_0 GGUF file")
    try:
        from realtimetts_breeze_native import executable_path
    except ImportError as exc:
        raise RuntimeError(
            "--model needs the optional realtimetts-breeze-native wheel; "
            "see docs/engines/breeze.md, or connect with --worker-url"
        ) from exc
    # Select both ports together; never reuse or terminate an existing worker.
    with ExitStack() as reserved:
        ports = []
        for _ in range(2):
            sock = reserved.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    http_port, ws_port = ports
    base_url, ws_url = f"http://127.0.0.1:{http_port}", f"ws://127.0.0.1:{ws_port}"
    voice_dir = voice_dir.expanduser().resolve()
    voice_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(executable_path()), str(model), "--host", "127.0.0.1",
         "--port", str(http_port), "--ws-port", str(ws_port),
         "--chunk-first", str(chunk_first), "--chunk-max", str(chunk_max), "--voices-dir", str(voice_dir)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + timeout
        with requests.Session() as session:
            session.trust_env = False
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Breeze worker exited with code {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Breeze worker did not become ready; see its startup output")
                try:
                    response = session.get(base_url + "/health", timeout=1)
                    response.raise_for_status()
                except requests.RequestException:
                    time.sleep(0.2)
                    continue
                health = response.json()
                if (health.get("native_checkpoints") is not True
                        or health.get("sample_rate") != 24000
                        or health.get("ws_port") != ws_port):
                    raise RuntimeError("Breeze worker has incompatible native controls or audio format")
                if not str(health.get("backend", "")).lower().startswith("vulkan"):
                    raise RuntimeError("Breeze could not select a Vulkan GPU; check the GPU driver. "
                                       "Use --worker-url for an explicitly configured CPU worker.")
                if process.poll() is not None:
                    raise RuntimeError("Breeze worker exited during startup")
                break
        yield base_url, ws_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


# Tune the actual worker before readiness. No trial requests run during user speech.
_PROFILES = ((2, 4), (4, 6), (4, 12), (4, 16))
_CALIBRATION_TEXT = (
    "We walked beside the river and watched the boats pass under the old stone bridge. "
    "After a while, we found a quiet place to sit and talk about our plans for the weekend. "
    "The afternoon sun was shining through the trees, and a gentle breeze moved across the water."
)


def _profile_key(model, voice, settings):
    """Cache only when the GPU and driver can be identified reliably."""
    import hashlib
    import json
    import os
    import platform
    from realtimetts_breeze_native import executable_path
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not gpu:
        return None
    model = Path(model)
    identity = {
        "schema": 1, "gpu": gpu, "platform": platform.platform(),
        "native": hashlib.sha256(Path(executable_path()).read_bytes()).hexdigest(),
        "model": [str(model.resolve()), model.stat().st_size, model.stat().st_mtime_ns],
        "reference": hashlib.sha256(Path(voice.ref_audio).read_bytes()).hexdigest(),
        "text": voice.ref_text, "settings": settings,
        "visible_devices": os.environ.get("GGML_VK_VISIBLE_DEVICES", ""),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _choose_profile(rows):
    # Reject a faster first packet if the rest of its stream starves playback.
    continuous = [row for row in rows if row["gap_ms"] <= 1 and row["rtf"] < 1]
    choices = continuous or rows
    return min(choices, key=lambda row: (
        0 if continuous else row["gap_ms"],
        row["first_pcm_ms"] if continuous else row["rtf"],
    ))


def _measure_profile(engine):
    class Sink:
        def __init__(self):
            self.started = time.monotonic()
            self.first = self.last = self.end = None
            self.audio = self.gap = 0.0
        def put(self, pcm, *args, **kwargs):
            now = time.monotonic() - self.started
            duration = len(pcm) / 48000
            if self.first is None:
                self.first = now
                self.end = now
            self.gap += max(0, now - self.end)
            self.end = max(now, self.end) + duration
            self.last = now
            self.audio += duration
    previous = engine.queue
    # A real warm request fills the reference-prefix cache and native shaders.
    sink = Sink()
    engine.queue = sink
    try:
        if not engine.synthesize("Hello, this is a short test."):
            raise RuntimeError("Breeze calibration warmup failed")
        sink = Sink()
        engine.queue = sink
        if not engine.synthesize(_CALIBRATION_TEXT) or not sink.audio:
            raise RuntimeError("Breeze calibration produced no audio")
        return dict(first_pcm_ms=sink.first * 1000, rtf=sink.last / sink.audio,
                    audio_s=sink.audio, gap_ms=sink.gap * 1000)
    finally:
        engine.queue = previous


@contextmanager
def calibrated_worker(model, voice_dir, voice, *, settings, retune=False):
    """Select measured streaming chunks, then run only the chosen worker."""
    import json
    from RealtimeTTS.engines.breeze_engine import BreezeEngine
    cache_path = Path(voice_dir).parent / "streaming-profile.json"
    key = _profile_key(model, voice, settings)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf8"))
        profile = tuple(cached["selected"]["profile"])
        if retune or key is None or cached["key"] != key or profile not in _PROFILES:
            profile = None
    except (OSError, ValueError, KeyError, TypeError):
        profile = None
    if profile is not None:
        print(f"Breeze streaming: cached measured chunks {profile[0]}/{profile[1]}", flush=True)
    else:
        rows = []
        # Start with the established cadence. Probe smaller chunks only if it
        # has headroom; probe larger batches only if it cannot keep up.
        candidates = [(4, 6)]
        while candidates:
            current = candidates.pop(0)
            print(f"Calibrating Breeze chunks {current[0]}/{current[1]} before readiness...", flush=True)
            with local_worker(model, voice_dir, chunk_first=current[0], chunk_max=current[1]) as urls:
                engine = BreezeEngine(base_url=urls[0], ws_url=urls[1], **settings)
                try:
                    engine.set_voice(voice)
                    row = _measure_profile(engine)
                finally:
                    engine.shutdown()
            row["profile"] = current
            rows.append(row)
            print("Breeze calibration: " + json.dumps(row), flush=True)
            if current == (4, 6):
                candidates.append((2, 4) if row["gap_ms"] <= 1 and row["rtf"] < .95 else (4, 12))
            elif current == (4, 12) and (row["gap_ms"] > 1 or row["rtf"] >= 1):
                candidates.append((4, 16))
        selected = _choose_profile(rows)
        profile = tuple(selected["profile"])
        print(f"Breeze streaming: selected measured chunks {profile[0]}/{profile[1]}", flush=True)
        if selected["gap_ms"] > 1 or selected["rtf"] >= 1:
            print("Breeze warning: this GPU/load did not sustain gap-free real-time playback in calibration.", flush=True)
        if key is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(key=key, selected=selected, measurements=rows), indent=2), encoding="utf8")
            temporary.replace(cache_path)
    with local_worker(model, voice_dir, chunk_first=profile[0], chunk_max=profile[1]) as urls:
        yield urls
