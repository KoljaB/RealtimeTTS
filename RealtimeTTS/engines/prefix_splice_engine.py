"""Opt-in, bounded speculative prefix synthesis for TextToAudioStream.

The first two complete words produce a first-word commit. Four words produce
the second commit through word three. A final bounded block completes the text.
Only retained PCM participates in a crossfade; committed PCM is immutable.
"""
from __future__ import annotations

import io
import http.client
import json
import math
import queue
import re
import socket
import threading
import time
from urllib.parse import urlsplit
import wave
from contextlib import contextmanager

import numpy as np

from .base_engine import BaseEngine
from .._audio_backend import pa
from ..prefix_splice import boundary_center, words


class QwenHttpSynthesizer:
    """Complete WAV and incremental PCM calls to an existing CPU Qwen server."""

    sample_rate = 24000

    def __init__(self, url, api_key, voice="mira_v5_spark_de", language="german", timeout=30):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.voice = voice
        self.language = language
        self.timeout = timeout
        self.capabilities = None
        self._network_lock = threading.Lock()
        self._socket = None
        self._cancel_event = threading.Event()

    def bind_cancel_event(self, event):
        self._cancel_event = event

    def cancel(self):
        with self._network_lock:
            sock = self._socket
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    @contextmanager
    def _response(self, path, payload=None):
        target = urlsplit(self.url)
        if target.scheme not in ("http", "https") or not target.hostname or target.username or target.query or target.fragment:
            raise ValueError("Expected an HTTP(S) endpoint without credentials, query or fragment")
        owner = self
        base = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
        class Connection(base):
            def connect(self):
                if owner._cancel_event.is_set():
                    raise _Cancelled()
                super().connect()
                with owner._network_lock:
                    if owner._cancel_event.is_set():
                        self.close()
                        raise _Cancelled()
                    owner._socket = self.sock
        headers = {"Authorization": "Bearer " + self.api_key}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode()
        connection = Connection(target.hostname, target.port, timeout=self.timeout)
        try:
            if self._cancel_event.is_set():
                raise _Cancelled()
            connection.request("POST" if payload is not None else "GET", target.path+path, body=data, headers=headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise RuntimeError(f"Qwen HTTP status {response.status}")
                yield response
        finally:
            with self._network_lock:
                self._socket = None
            connection.close()

    def _request(self, path, payload=None):
        with self._response(path, payload) as response:
            return response.read()

    def stream(self, text):
        """Yield PCM as bytes arrive, preserving odd-byte network splits."""
        payload = {"input": text, "voice": self.voice, "language": self.language,
                   "response_format": "pcm", "seed": 42, "do_sample": False,
                   "subtalker_do_sample": False}
        pending = b""
        with self._response("/v1/audio/speech", payload) as response:
            if response.getheader("Content-Type", "").split(";")[0] != "audio/pcm":
                raise ValueError("Expected raw PCM streaming response")
            expected = {"X-Audio-Sample-Rate": str(self.sample_rate), "X-Audio-Channels": "1",
                        "X-Audio-Bits-Per-Sample": "16", "X-Audio-Encoding": "s16le"}
            if any(response.getheader(name) != value for name, value in expected.items()):
                raise ValueError("Unexpected streaming PCM metadata")
            while not self._cancel_event.is_set():
                try:
                    chunk = response.read1(8192)
                except (OSError, http.client.HTTPException):
                    if self._cancel_event.is_set():
                        raise _Cancelled() from None
                    raise
                if not chunk:
                    break
                pending += chunk
                count = len(pending)//2*2
                if count:
                    yield np.frombuffer(pending[:count], dtype="<i2").astype(np.float32)/32768
                    pending = pending[count:]
        if self._cancel_event.is_set():
            raise _Cancelled()
        if pending:
            raise ValueError("Truncated PCM sample")

    def prepare(self):
        self.capabilities = json.loads(self._request("/v1/capabilities"))
        engine = self.capabilities["engine"]
        if engine.get("device") != "cpu" or not engine.get("cpu_only"):
            raise ValueError("This backend requires the CPU-only Qwen server")

    def __call__(self, text):
        payload = {"input": text, "voice": self.voice, "language": self.language,
                   "response_format": "wav", "seed": 42, "do_sample": False,
                   "subtalker_do_sample": False}
        raw = self._request("/v1/audio/speech", payload)
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, self.sample_rate):
                raise ValueError("Unexpected Qwen WAV format")
            result = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)/32768
        if not len(result) or not np.any(result):
            raise ValueError("Empty/silent Qwen synthesis")
        return result


class _Cancelled(Exception):
    pass


class PrefixSpliceEngine(BaseEngine):
    """Experimental generator engine, with injected resident aligner/backend.

    ``synthesizer(text)`` returns mono float PCM at ``sample_rate``; ``aligner``
    implements ``align(audio, rate, text)``. Prepare expensive models BEFORE play.
    When the backend offers stream() and the aligner align_prefix(), following
    stages commit stable boundaries before EOF. Initial streaming is optional.
    """

    def __init__(self, synthesizer, aligner, sample_rate=24000, commit_words=(1, 3),
                 crossfade_ms=10, boundary_search_ms=5, max_block_words=24,
                 max_block_chars=240, idle_flush_seconds=.35, max_queue_seconds=3,
                 streaming=True, alignment_interval_seconds=.16,
                 right_context_seconds=.12, min_anchor_confidence=.25,
                 stream_initial_prefix=False, continuous_playback=False):
        self.synthesizer = synthesizer
        self.aligner = aligner
        self.sample_rate = sample_rate
        self.commit_words = tuple(commit_words)
        self.crossfade_ms = crossfade_ms
        self.boundary_search_ms = boundary_search_ms
        self.max_block_words = max_block_words
        self.max_block_chars = max_block_chars
        self.idle_flush_seconds = idle_flush_seconds
        self.max_queue_seconds = max_queue_seconds
        self.streaming = streaming
        self.stream_initial_prefix = stream_initial_prefix
        self.continuous_playback = continuous_playback
        self._pending_audio = []
        self._pending_samples = 0
        self.alignment_interval_seconds = alignment_interval_seconds
        self.right_context_seconds = right_context_seconds
        self.min_anchor_confidence = min_anchor_confidence
        if alignment_interval_seconds <= 0 or right_context_seconds < .04 or not 0 <= min_anchor_confidence <= 1:
            raise ValueError("Invalid streaming alignment gate configuration")
        if not self.commit_words or any(not isinstance(x, int) or x < 1 for x in self.commit_words) or tuple(sorted(set(self.commit_words))) != self.commit_words:
            raise ValueError("commit_words must be increasing positive word counts")
        if max_block_words <= self.commit_words[-1] or max_block_chars < 32:
            raise ValueError("Block limits must accommodate all prefix stages")
        if not 0 <= boundary_search_ms <= 5 or not 0 < crossfade_ms <= 40:
            raise ValueError("Use a 0..5 ms boundary correction and 0..40 ms crossfade")
        if min(sample_rate, idle_flush_seconds, max_queue_seconds) <= 0:
            raise ValueError("Rates, timeouts and queue bounds must be positive")
        self._guard = threading.RLock()
        self._running = False
        self._cancel = threading.Event()
        self._threads = []
        self.metrics = []
        self.last_error = None

    def post_init(self):
        self.engine_name = "prefix_splice"
        self.can_consume_generators = True

    def get_stream_info(self):
        return pa.paInt16, 1, self.sample_rate

    def _event(self, event, **fields):
        self.metrics.append({"event": event, "seconds": time.perf_counter()-self._started, **fields})

    def _operation(self, function):
        """Cancel waiting promptly; late worker results never touch playback."""
        result = queue.Queue(maxsize=1)
        def run():
            try:
                result.put((True, function()))
            except BaseException as error:
                result.put((False, error))
        worker = threading.Thread(target=run, daemon=True, name="prefix-splice-operation")
        with self._guard:
            if self._cancel.is_set():
                raise _Cancelled()
            self._threads.append(worker)
            worker.start()
        while not self._cancel.is_set():
            try:
                ok, value = result.get(timeout=.02)
                worker.join(.05)
                if self._cancel.is_set():
                    raise _Cancelled()
                if not ok:
                    raise value
                return value
            except queue.Empty:
                pass
        raise _Cancelled()

    def _render(self, text, need_alignment):
        self._event("synthesis_start", text=text)
        start = time.perf_counter()
        audio = np.asarray(self._operation(lambda: self.synthesizer(text)), dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Invalid synthesized PCM")
        self._event("synthesis_end", duration_seconds=len(audio)/self.sample_rate,
                    elapsed_seconds=time.perf_counter()-start,
                    rtf=(time.perf_counter()-start)/(len(audio)/self.sample_rate))
        aligned = None
        if need_alignment:
            start = time.perf_counter()
            aligned = self._operation(lambda: self.aligner.align(audio, self.sample_rate, text))
            if [w["word"] for w in aligned["words"]] != words(text):
                raise ValueError("Alignment transcript mismatch")
            end = 0
            for word in aligned["words"]:
                if not end <= word["start"] < word["end"] <= len(audio)/self.sample_rate:
                    raise ValueError("Invalid alignment boundary")
                end = word["end"]
            self._event("alignment_end", elapsed_seconds=time.perf_counter()-start)
        return audio, aligned

    def _center(self, audio, aligned, count):
        timings = aligned["words"]
        n = round(self.sample_rate*self.crossfade_ms/1000)
        center, detail = boundary_center(audio, self.sample_rate, timings[count-1]["end"],
                                         timings[count]["start"], n, self.boundary_search_ms)
        return center, detail

    def _emit(self, audio):
        if self.continuous_playback:
            if self._cancel.is_set():
                raise _Cancelled()
            if len(audio):
                if not self._pending_audio:
                    self._event("first_buffer_commit", samples=len(audio))
                self._pending_samples += len(audio)
                if self._pending_samples > self.sample_rate * 60:
                    raise ValueError("Buffered block exceeds 60 seconds")
                self._pending_audio.append(np.asarray(audio, dtype=np.float32).copy())
            return
        self._enqueue(audio)

    def _flush_block(self):
        """Release only a completed block: inference cannot starve it mid-word."""
        if self._pending_audio:
            audio = np.concatenate(self._pending_audio)
            self._pending_audio = []
            self._pending_samples = 0
            self._event("playback_block_ready", duration_seconds=len(audio)/self.sample_rate)
            self._enqueue(audio)

    def _enqueue(self, audio):
        # Queue at most max_queue_seconds of fixed-size PCM chunks. Slow playback
        # applies backpressure rather than growing an unbounded PCM backlog.
        chunk = max(1, round(self.sample_rate*.02))
        limit = max(1, math.ceil(self.max_queue_seconds/.02))
        for offset in range(0, len(audio), chunk):
            while self.queue.qsize() >= limit:
                if self._cancel.wait(.01):
                    raise _Cancelled()
            data = np.clip(np.rint(audio[offset:offset+chunk]*32768), -32768, 32767).astype("<i2").tobytes()
            with self._guard:
                if self._cancel.is_set():
                    raise _Cancelled()
                self.queue.put_nowait(data)
                now = time.perf_counter()
                if self._first_commit and now > self._available_until+.005:
                    self._event("estimated_underrun", gap_seconds=now-self._available_until)
                self._available_until = max(now, self._available_until)+len(data)/(2*self.sample_rate)
                if not self._first_commit:
                    self._event("first_queue_commit", samples=len(data)//2)
                    self._first_commit = True

    def _stage(self, text, count=None):
        self._produce_stage(text, count)
        if count is None:
            self._flush_block()

    def _produce_stage(self, text, count=None):
        """Hand off at the OLD anchor, then retain the NEW anchor if requested."""
        if self._retained is not None and text == self._retained["text"] and count is None:
            old = self._retained
            self._emit(old["audio"][old["start"]:])
            self._retained = None
            return
        if (self.streaming and (self._retained is not None or self.stream_initial_prefix)
                and hasattr(self.synthesizer, "stream") and hasattr(self.aligner, "align_prefix")):
            return self._stream_stage(text, count)
        audio, aligned = self._render(text, count is not None or self._retained is not None)
        n = round(self.sample_rate*self.crossfade_ms/1000)
        begin = 0
        transition = None
        if self._retained is not None:
            old = self._retained
            center, detail = self._center(audio, aligned, old["count"])
            right_start = center-n//2
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            transition = old["audio"][old["start"]:old["start"]+n]*(1-ramp) + audio[right_start:right_start+n]*ramp
            begin = right_start+n
            self._event("merge", after_word=old["count"], left=old["detail"], right=detail, overlap_samples=n)
        stop = len(audio)
        retained = None
        if count is not None:
            center, detail = self._center(audio, aligned, count)
            stop = center-n//2
            if stop < begin:
                raise ValueError("Prefix boundaries overlap; refusing to duplicate committed PCM")
            retained = {"audio": audio, "start": stop, "count": count, "detail": detail, "text": text}
        # Validate BOTH new boundaries before sending anything from this stage.
        if transition is not None:
            self._emit(transition)
        self._emit(audio[begin:stop])
        self._retained = retained

    def _stream_stage(self, text, count):
        """Consume PCM concurrently; commit a stable handoff before network EOF."""
        self._event("synthesis_start", text=text, transport="pcm_stream")
        started = time.perf_counter()
        packets = queue.Queue(maxsize=8)
        cancel = self._cancel
        stream_stop = threading.Event()
        def send(item):
            while not cancel.is_set() and not stream_stop.is_set():
                try:
                    packets.put(item, timeout=.02)
                    return
                except queue.Full:
                    pass
        def receive():
            try:
                for chunk in self.synthesizer.stream(text):
                    if cancel.is_set() or stream_stop.is_set():
                        return
                    send(("pcm", chunk))
                send(("end", time.perf_counter()))
            except BaseException as error:
                send(("error", error))
        reader = threading.Thread(target=receive, daemon=True, name="prefix-splice-pcm")
        with self._guard:
            if cancel.is_set():
                raise _Cancelled()
            self._threads.append(reader)
            reader.start()
        old = self._retained
        old_done = old is None
        new_done = count is None
        buffer = np.empty(0, dtype=np.float32)
        chunks = []
        total = sent = 0
        n = round(self.sample_rate*self.crossfade_ms/1000)
        previous = {}
        last_attempt = 0
        def commit(aligned, anchor, is_old):
            nonlocal old_done, new_done, sent
            center, detail = self._center(buffer, aligned, anchor)
            if is_old:
                right = center-n//2
                ramp = np.linspace(0, 1, n, dtype=np.float32)
                mix = old["audio"][old["start"]:old["start"]+n]*(1-ramp)+buffer[right:right+n]*ramp
                # The old reserve must never be replayed after this handoff,
                # including if a later network/alignment operation fails.
                self._retained = None
                self._event("merge", after_word=anchor, left=old["detail"], right=detail,
                            overlap_samples=n, streamed=True, received_samples=total)
                self._emit(mix)
                sent = right+n
                old_done = True
                if count is not None:
                    safe_end = round(aligned["words"][anchor]["end"]*self.sample_rate)
                    self._emit(buffer[sent:max(sent, safe_end)])
                    sent = max(sent, safe_end)
            else:
                stop = center-n//2
                if stop < sent:
                    raise ValueError("Streaming boundaries overlap committed audio")
                self._emit(buffer[sent:stop])
                sent = stop
                self._retained = {"audio": buffer, "start": stop, "count": count,
                                  "detail": detail, "text": text, "complete": False}
                new_done = True
            self._event("stream_boundary_commit", after_word=anchor, received_samples=total)
        def candidate(anchor):
            try:
                aligned = self._operation(lambda: self.aligner.align_prefix(buffer, self.sample_rate, text, anchor))
                timings = aligned["words"]
                if [w["word"] for w in timings] != words(text)[:anchor+1]:
                    raise ValueError("Streaming transcript mismatch")
                end = 0
                for w in timings:
                    if not end <= w["start"] < w["end"] <= len(buffer)/self.sample_rate:
                        raise ValueError("Invalid partial alignment")
                    end = w["end"]
                left = [p for p in aligned["phones"] if p["word_index"] == anchor-1][-1]
                right = [p for p in aligned["phones"] if p["word_index"] == anchor][0]
                signature = (timings[anchor-1]["end"], timings[anchor]["start"])
                if min(left["score"], right["score"]) < self.min_anchor_confidence or end+self.right_context_seconds > len(buffer)/self.sample_rate:
                    previous.pop(anchor, None)
                    return None
                self._center(buffer, aligned, anchor)  # validate full overlap now
                ready = previous.get(anchor) == signature
                previous[anchor] = signature
                self._event("stream_alignment_candidate", after_word=anchor, stable=ready,
                            boundary_seconds=list(signature), received_samples=total)
                return aligned if ready else None
            except ValueError:
                previous.pop(anchor, None)
                return None  # incomplete evidence: collect more PCM, then retry
        try:
            while not cancel.is_set():
                try:
                    kind, value = packets.get(timeout=.02)
                except queue.Empty:
                    continue
                if kind == "error":
                    if cancel.is_set():
                        raise _Cancelled()
                    raise value
                if kind == "end":
                    if not total:
                        raise ValueError("Empty PCM stream")
                    if chunks:
                        buffer = np.concatenate(chunks)
                    self._event("synthesis_end", streamed=True, duration_seconds=total/self.sample_rate,
                                elapsed_seconds=value-started, rtf=(value-started)/(total/self.sample_rate))
                    if not old_done or not new_done:
                        self._event("stream_boundary_fallback", reason="EOF before stable partial boundary")
                        aligned = self._operation(lambda: self.aligner.align(buffer, self.sample_rate, text))
                        if [w["word"] for w in aligned["words"]] != words(text):
                            raise ValueError("Alignment transcript mismatch")
                        end = 0
                        for word in aligned["words"]:
                            if not end <= word["start"] < word["end"] <= len(buffer)/self.sample_rate:
                                raise ValueError("Invalid alignment boundary")
                            end = word["end"]
                        if not old_done:
                            commit(aligned, old["count"], True)
                        if not new_done:
                            commit(aligned, count, False)
                    if count is None and len(buffer):
                        self._emit(buffer[sent:])
                    if count is not None and self._retained is not None:
                        self._retained["audio"] = buffer
                        self._retained["complete"] = True
                    return
                chunk = np.asarray(value, dtype=np.float32)
                if chunk.ndim != 1 or not np.isfinite(chunk).all():
                    raise ValueError("Invalid PCM chunk")
                if not len(chunk):
                    continue
                if total == 0:
                    self._event("first_pcm_received")
                total += len(chunk)
                if total > self.sample_rate*60:
                    raise ValueError("PCM block exceeds 60-second safety limit")
                if old_done and count is None:
                    self._emit(chunk)
                    continue
                chunks.append(chunk)
                if total-last_attempt >= round(self.alignment_interval_seconds*self.sample_rate) and (not old_done or not new_done):
                    buffer = np.concatenate(chunks)
                    last_attempt = total
                    if not old_done:
                        aligned = candidate(old["count"])
                        if aligned is not None:
                            commit(aligned, old["count"], True)
                    if old_done and not new_done:
                        aligned = candidate(count)
                        if aligned is not None:
                            commit(aligned, count, False)
                if old_done and count is None:
                    self._emit(buffer[sent:])
                    buffer = np.empty(0, dtype=np.float32)
                    chunks = []
                    sent = 0
            raise _Cancelled()
        finally:
            stream_stop.set()
            if reader.is_alive() and hasattr(self.synthesizer, "cancel"):
                self.synthesizer.cancel()
            reader.join(.05)

    def synthesize(self, source, sentence_count=0):
        with self._guard:
            if self._running or any(t.is_alive() for t in self._threads):
                raise RuntimeError("Previous synthesis/input worker has not finished")
            self._running = True
            self._threads = []
            self._cancel = threading.Event()
            if hasattr(self.synthesizer, "bind_cancel_event"):
                self.synthesizer.bind_cancel_event(self._cancel)
            self._retained = None
            self._pending_audio = []
            self._pending_samples = 0
            self._first_commit = False
            self._available_until = 0
            self._started = time.perf_counter()
            self.metrics = []
            self.last_error = None
        incoming = queue.Queue(maxsize=64)
        cancel = self._cancel
        def send(item):
            while not cancel.is_set():
                try:
                    incoming.put(item, timeout=.02)
                    return
                except queue.Full:
                    pass
        def read():
            # A token is complete ONLY after a delimiter or input exhaustion.
            token = ""
            try:
                for piece in source:
                    if not isinstance(piece, str):
                        raise TypeError("Text iterator must yield strings")
                    for char in piece:
                        if cancel.is_set():
                            return
                        if char.isspace() or char in ".!?;,:":
                            if token:
                                send(("word", token + (char if not char.isspace() else "")))
                                token = ""
                            elif char in ".!?;,:":
                                send(("boundary", char))
                        else:
                            token += char
                            if len(token) > self.max_block_chars:
                                raise ValueError("A single token exceeds max_block_chars")
                if token:
                    send(("word", token))
                send(("end", None))
            except BaseException as error:
                send(("error", error))
        reader = threading.Thread(target=read, daemon=True, name="prefix-splice-input")
        self._threads.append(reader)
        reader.start()
        block = []
        stage = 0
        last_input = time.perf_counter()
        try:
            while not cancel.is_set():
                try:
                    kind, value = incoming.get(timeout=.02)
                except queue.Empty:
                    if self._retained is not None and block and time.perf_counter()-last_input >= self.idle_flush_seconds:
                        self._stage(" ".join(block))
                        block, stage = [], 0
                    continue
                last_input = time.perf_counter()
                if kind == "error":
                    raise value
                if kind == "end":
                    if block:
                        self._stage(" ".join(block))
                    return True
                if kind == "boundary":
                    if block:
                        self._stage(" ".join(block)+value)
                        block, stage = [], 0
                    continue
                if len(words(value)) != 1 or re.search(r"\d", value):
                    raise ValueError("Provide normalized, spelled-out words")
                if block and len(" ".join(block))+len(value)+1 > self.max_block_chars:
                    self._stage(" ".join(block))
                    block, stage = [], 0
                block.append(value)
                if value[-1:] in ".!?;,:" or len(block) >= self.max_block_words:
                    self._stage(" ".join(block))
                    block, stage = [], 0
                elif stage < len(self.commit_words) and len(block) == self.commit_words[stage]+1:
                    self._stage(" ".join(block), self.commit_words[stage])
                    stage += 1
            return False
        except _Cancelled:
            return False
        except BaseException as error:
            self.last_error = error
            # On failure, play the uncommitted portion of the most recent valid
            # prefix once. No replay of the already committed first word.
            if not cancel.is_set() and self._retained is not None and self._retained.get("complete", True):
                self._event("retained_tail_recovery", error_type=type(error).__name__)
                self._emit(self._retained["audio"][self._retained["start"]:])
            raise
        finally:
            cancel.set()
            reader.join(.05)
            with self._guard:
                self._retained = None
                self._pending_audio = []
                self._pending_samples = 0
                self._running = False

    def stop(self):
        with self._guard:
            self._cancel.set()
            if hasattr(self.synthesizer, "cancel"):
                self.synthesizer.cancel()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def shutdown(self):
        self.stop()

    def get_voices(self):
        return [getattr(self.synthesizer, "voice", "configured")]

    def set_voice(self, voice):
        if self._running:
            raise RuntimeError("Cannot change voice during synthesis")
        self.synthesizer.voice = str(voice)

    def set_voice_parameters(self, **kwargs):
        raise NotImplementedError("Configure the synthesizer before starting")
