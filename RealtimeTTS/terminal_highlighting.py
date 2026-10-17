"""Terminal word/character highlighting from measured alignment and the audio DAC clock.

Requires rich, pyaudio and the alignment-cpu dependencies. This module only handles
alignment and playback; it neither starts servers nor writes files.
"""
from collections import deque
import os
from queue import Empty, Queue
import re
from threading import Event, Lock, Thread
import time

import numpy as np
import pyaudio
from rich.console import Console
from rich.live import Live
from rich.text import Text

from .alignment.base import CharacterAlignment
from .alignment.ctc import ctc_forced_align
from .alignment.onnx_ctc import OnnxCTCBackend
from .alignment.timeline import AlignmentConfig

_VOCAL_CUES = re.compile(r"\((?:laugh|cough|clears throat|sigh)\)", re.IGNORECASE)


class _CompletedCTCAligner:
    """Wait for a complete fragment; never extrapolate character timestamps."""
    supports_partial = False

    def __init__(self, model_dir):
        self._ctc = OnnxCTCBackend(model_dir, threads=2)

    def align_characters(self, text, audio, sample_rate):
        if not text.strip():
            return []
        normalized, targets = self._ctc.encode(text)
        spans = ctc_forced_align(
            self._ctc.emissions(audio, sample_rate), targets, blank_id=self._ctc.blank_id,
        )
        return [
            CharacterAlignment(normalized[s.token_index],
                               s.start_frame * self._ctc.frame_seconds,
                               s.end_frame * self._ctc.frame_seconds,
                               s.token_index, float(np.exp(s.score)))
            for s in spans
        ]


class _PCMPlayback:
    def __init__(self, finished, rate):
        self.queue, self.finished, self.rate = Queue(), finished, rate
        self.pending, self.samples, self.schedule = b"", 0, deque()

    def callback(self, _input, frames, timing, _status):
        result = bytearray()
        while len(result) < frames * 2:
            if not self.pending:
                try:
                    self.pending = self.queue.get_nowait()
                except Empty:
                    break
            count = min(len(self.pending), frames * 2 - len(result))
            result.extend(self.pending[:count])
            self.pending = self.pending[count:]
        count = len(result) // 2
        if count:
            self.schedule.append((timing["output_buffer_dac_time"], self.samples, count))
            self.samples += count
        complete = self.finished.is_set() and not self.pending and self.queue.empty()
        result.extend(bytes(frames * 2 - len(result)))
        return bytes(result), pyaudio.paComplete if complete else pyaudio.paContinue

    def position(self, device_time):
        while len(self.schedule) > 1 and self.schedule[1][0] <= device_time:
            self.schedule.popleft()
        if self.schedule:
            at, start, count = self.schedule[0]
            offset = (device_time - at) * self.rate
            if 0 <= offset < count:
                return (start + offset) / self.rate
        return None  # No highlight before playback or during an audio underrun.


class TerminalHighlighter:
    """Pass alignment/update to TextToAudioStream; input_text optionally shows its markup."""

    def __init__(self, model_dir, *, input_text=None):
        self.input_text = input_text
        self._input_plain, self._input_runs = "", []
        self.alignment = AlignmentConfig(
            source="model", mode="synced", level="character", language="auto",
            aligner=_CompletedCTCAligner(model_dir), max_buffer_seconds=10,
            update_interval=0.32, max_pending=32,
        )
        self._latest, self._spoken, self._lock = None, 0, Lock()
        self._final = Event()

    def _prepare_input(self, stream):
        """Use the synthesis grammar; keep instructions out of spoken-text offsets."""
        position, plain_length = 0, 0
        pieces, self._input_runs = [], []

        def characters():
            nonlocal position
            for position, character in enumerate(self.input_text, 1):
                yield character

        parser = stream._new_inline_markup_parser()
        for kind, value in parser.iter_actions(characters(), coalesce_text=True):
            if kind != "text":
                continue
            start = position - len(value)
            if self.input_text[start:position] != value:
                raise ValueError("Cannot map this markup to the original input text.")
            if self._input_runs and self._input_runs[-1][1] == start:
                self._input_runs[-1][1] = position
            else:
                self._input_runs.append([start, position, plain_length])
            pieces.append(value)
            plain_length += len(value)
        self._input_plain = "".join(pieces)

    def _with_instructions(self, styled):
        visible, cursor = Text(), 0
        for start, end, plain_start in self._input_runs:
            visible.append(self.input_text[cursor:start], style="dim")
            visible.append(styled[plain_start:plain_start + end - start])
            cursor = end
        visible.append(self.input_text[cursor:], style="dim")
        return visible

    def update(self, snapshot):
        with self._lock:
            if self._latest is None or snapshot.revision >= self._latest.revision:
                self._latest = snapshot
                if snapshot.final:
                    self._final.set()

    def _render(self, seconds, complete=False):
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return (self._with_instructions(Text(self._input_plain, style="white"))
                    if self.input_text is not None else Text(""))
        if self.input_text is not None and (
                not self._input_plain.startswith(snapshot.text)
                or (complete and snapshot.text != self._input_plain)):
            raise ValueError("input_text does not match the text fed to the stream.")
        styled = Text(self._input_plain if self.input_text is not None else snapshot.text,
                      style="white")
        if seconds is not None:
            self._spoken = max(
                (s.text_end for s in snapshot.words
                 if s.end_time <= seconds and s.text_end is not None), default=self._spoken,
            )
        styled.stylize("#667080", 0, len(styled) if complete else self._spoken)
        if seconds is not None and not complete:
            for spans, style in ((snapshot.words, "bold #58cbda"),
                                 (snapshot.characters, "underline #eff7f9")):
                for span in spans:
                    if (span.start_time <= seconds < span.end_time and span.text.strip()
                            and span.text_start is not None and span.text_end is not None):
                        styled.stylize(style, span.text_start, span.text_end)
                        break
        if self.input_text is not None:
            return self._with_instructions(styled)
        # Slice after styling so hidden vocal cues cannot shift measured ranges.
        visible, cursor = Text(style=styled.style), 0
        for match in _VOCAL_CUES.finditer(styled.plain):
            start, end = match.span()
            if end == len(styled) or styled.plain[end].isspace():
                while start > cursor and styled.plain[start - 1] in " \t":
                    start -= 1
            visible.append(styled[cursor:start])
            cursor = end
        visible.append(styled[cursor:])
        return visible

    def play(self, stream, **play_options):
        """Play mono PCM16; highlight only actual, currently audible span intervals."""
        console = Console()
        if not console.file.isatty():
            raise RuntimeError("Run the demo in a terminal without redirecting stdout.")
        stream_format, channels, rate = stream.engine.get_stream_info()
        if stream_format != pyaudio.paInt16 or channels != 1:
            raise ValueError("TerminalHighlighter requires mono PCM16 output.")
        if stream.alignment is None or stream.alignment.config is not self.alignment:
            raise ValueError("Pass display.alignment and display.update to TextToAudioStream.")
        with self._lock:
            self._latest, self._spoken = None, 0
            self._final.clear()
        if self.input_text is not None:
            self._prepare_input(stream)
        finished, failures = Event(), []
        playback = _PCMPlayback(finished, rate)

        def generate():
            try:
                stream.play(muted=True, on_audio_chunk=playback.queue.put, **play_options)
                if not self._final.wait(timeout=10):
                    raise TimeoutError("Alignment did not finish; playback is not fully aligned.")
            except Exception as exc:
                failures.append(exc)
            finally:
                finished.set()

        worker = Thread(target=generate, name="terminal-tts", daemon=True)
        audio, device = pyaudio.PyAudio(), None
        old_term = os.environ.get("TERM")
        try:
            # Rich suppresses animation for TERM=dumb, even in a real Windows terminal.
            if old_term in (None, "", "dumb", "unknown"):
                os.environ["TERM"] = "xterm-256color"
            console = Console(no_color=False, color_system="truecolor", force_interactive=True)
            device = audio.open(format=stream_format, channels=channels, rate=rate, output=True,
                                frames_per_buffer=max(1, rate // 100),
                                stream_callback=playback.callback, start=False)
            console.print("Start")
            worker.start()
            with Live(self._render(None), console=console, auto_refresh=False) as live:
                while not playback.pending:
                    try:
                        playback.pending = playback.queue.get(timeout=0.02)
                    except Empty:
                        if finished.is_set():
                            raise failures[0] if failures else RuntimeError("No audio produced.")
                device.start_stream()
                while device.is_active():
                    live.update(self._render(playback.position(device.get_time())), refresh=True)
                    time.sleep(1 / 60)
                # paComplete can arrive before the last queued hardware sample is audible.
                if playback.schedule:
                    at, _, count = playback.schedule[-1]
                    end = at + count / rate
                    deadline = time.monotonic() + max(0, end - device.get_time()) + 1
                    while device.get_time() < end:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Audio device clock stopped before playback ended.")
                        live.update(self._render(playback.position(device.get_time())), refresh=True)
                        time.sleep(1 / 60)
                if failures:
                    raise failures[0]
                snapshot = self._latest
                if snapshot is None or not snapshot.final or snapshot.coverage < 1.0 or snapshot.errors:
                    raise RuntimeError("Alignment is incomplete; not all speech could be highlighted.")
                live.update(self._render(None, complete=True), refresh=True)
        finally:
            if device is not None:
                device.close()
            audio.terminate()
            if worker.is_alive():
                stream.stop()
                worker.join(timeout=5)
            if old_term is None:
                os.environ.pop("TERM", None)
            else:
                os.environ["TERM"] = old_term
            if worker.is_alive():
                raise RuntimeError("Synthesis did not stop after closing terminal playback.")
