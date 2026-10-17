import io
import threading
import wave

import numpy as np
import pytest

from RealtimeTTS import TextToAudioStream, TimingInfo
from RealtimeTTS.audio_tempo import SpeedChange
from RealtimeTTS._audio_backend import pyaudio
from RealtimeTTS.alignment import FakeWordAligner, WordAlignment
import RealtimeTTS.text_to_stream as text_module
import RealtimeTTS.stream_player as player_module
from tests.test_text_to_stream_alignment import _PCMEngine, _FakePyAudio


class RecordingEngine(_PCMEngine):
    def post_init(self):
        super().post_init()
        self.spoken = []

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        self.spoken.append(text)
        return super().synthesize(text, sentence_count)


class WholeRunSplitter:
    @staticmethod
    def generate_sentences(source, **options):
        value = "".join(getattr(chunk, "text", chunk) for chunk in source)
        if value:
            yield value


@pytest.fixture
def stream_factory(monkeypatch):
    monkeypatch.setattr(player_module.pyaudio, "PyAudio", _FakePyAudio)
    monkeypatch.setattr(text_module, "_get_stream2sentence", lambda: WholeRunSplitter)

    def make(engine=None, **options):
        return TextToAudioStream(engine or RecordingEngine(), tokenizer="rule-based",
                                 playout_chunk_size=640, **options)
    return make


def collect_actions(stream, chunks):
    actions = list(stream._iter_inline_actions(iter(chunks)))
    controls = [(kind, value) for kind, value in actions if kind != "text"]
    text = "".join(value for kind, value in actions if kind == "text")
    return text, controls


def test_speed_tags_are_incremental_and_reset_to_constructor_default(stream_factory):
    stream = stream_factory(speed=1.25, speed_tags=True)
    source = "Before [speed=0.5]after [speed=reset]done"
    text, controls = collect_actions(stream, iter(source))
    assert text == "Before after done"
    assert controls == [("speed", 0.5), ("speed", 1.25)]


@pytest.mark.parametrize("value", ["0", "-1", "9", "nan", "inf", "bad", ""])
def test_invalid_speed_tags_remain_literal(stream_factory, value):
    stream = stream_factory(speed_tags=True)
    source = "hello [speed=" + value + "] world"
    assert collect_actions(stream, iter(source)) == (source, [])


def test_disabled_and_incomplete_tags_remain_literal(stream_factory):
    source = "hello [speed=0.5] world [speed=2"
    assert collect_actions(stream_factory(), iter(source)) == (source, [])
    source = "hello [speed=2"
    assert collect_actions(stream_factory(speed_tags=True), iter(source)) == (source, [])
    source = "[" + "a" * 300 + "]"
    assert collect_actions(stream_factory(speed_tags=True), iter(source)) == (source, [])


def test_speed_uses_custom_delimiters_and_preserves_other_inline_controls(stream_factory):
    stream = stream_factory(speed_tags=True)
    stream.add_voice("voice", "test-voice").add_pause("pause", 0.1)
    stream.set_voice_tag_delimiters("##", "##")
    text, controls = collect_actions(stream, iter("A##speed=2##B##voice####pause=0.2##C"))
    assert text == "ABC"
    assert controls == [("speed", 2.0), ("voice", ("voice", "test-voice")),
                        ("pause", ("pause", 0.2))]



def test_scoped_speed_restores_previous_rate_in_audio_and_wav(stream_factory, tmp_path):
    stream = stream_factory(speed_tags=True)
    received = []
    output = tmp_path / "scoped.wav"
    source = "One <speed factor='0.5'>Two <SR=2>Three[/speed]Four</speed>Five"
    stream.feed(iter(source)).play(output_wavfile=str(output), on_audio_chunk=received.append)
    assert [text.strip() for text in stream.engine.spoken] == ["One", "Two", "Three", "Four", "Five"]
    segments = [segment for segment in stream.tempo_timeline.segments if segment.closed]
    assert [segment.speed for segment in segments] == [1.0, 0.5, 2.0, 0.5, 1.0]
    assert [segment.source_duration for segment in segments] == pytest.approx([1.0] * 5)
    assert [segment.output_duration for segment in segments] == pytest.approx(
        [1.0, 2.0, 0.5, 2.0, 1.0], abs=0.12)
    with wave.open(str(output), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == b"".join(received)


def test_generator_engine_restores_nested_reset_scopes(stream_factory):
    engine = RecordingEngine()
    engine.can_consume_generators = True
    stream = stream_factory(engine, speed=1.25, speed_tags=True)
    stream.feed(iter("One <sr='.5'>Two [speed=reset]Three</speed>Four[/sr]Five")).play()
    assert [text.strip() for text in engine.spoken] == ["One", "Two", "Three", "Four", "Five"]
    assert [s.speed for s in stream.tempo_timeline.segments if s.closed] == [
        1.25, 0.5, 1.25, 0.5, 1.25]


def test_registered_style_wrap_restores_public_engine_voice(stream_factory):
    class VoiceEngine(RecordingEngine):
        def post_init(self):
            super().post_init()
            self.current_voice = "normal"
            self.spoken_voices = []

        def set_voice(self, voice):
            self.current_voice = voice

        def synthesize(self, text, sentence_count=0):
            self.spoken_voices.append(self.current_voice)
            return super().synthesize(text, sentence_count)

    engine = VoiceEngine()
    stream = stream_factory(engine)
    stream.add_emotion("whispering", "quiet").add_voice("happy", "happy")
    stream.feed(iter(
        "A <style name='Whispering'>B [happy]C</emotion>D[/WHISPERING]E"
    )).play()
    assert [text.strip() for text in engine.spoken] == list("ABCDE")
    assert engine.spoken_voices == ["normal", "quiet", "happy", "quiet", "normal"]
    assert engine.current_voice == "normal"


def test_registered_point_voice_establishes_scope_baseline(stream_factory):
    engine = RecordingEngine()
    voices = []
    engine.set_voice = voices.append
    stream = stream_factory(engine)
    stream.add_voice("normal", "normal-voice").add_emotion("whispering", "quiet-voice")
    stream.feed("[normal]A <whispering>B</style>C").play()
    assert voices == ["normal-voice", "quiet-voice", "normal-voice"]


def test_stream_parser_keeps_scope_across_internal_play_continuations(stream_factory):
    stream = stream_factory(speed_tags=True)
    stream.is_playing_flag = True
    try:
        assert collect_actions(stream, ["A<speed=.5>B"])[1] == [("speed", 0.5)]
        assert collect_actions(stream, ["</sr>C"])[1] == [("speed", 1.0)]
    finally:
        stream.is_playing_flag = False
    assert stream._extract_inline_actions("A[sr=.5]B</speed>C") == [
        ("text", "A"), ("speed", 0.5), ("text", "B"), ("speed", 1.0), ("text", "C")]


def test_full_play_preserves_audio_order_and_exports_processed_pcm(stream_factory, tmp_path):
    stream = stream_factory(speed_tags=True)
    received = []
    output = tmp_path / "tempo.wav"
    stream.feed(iter("One [speed=0.5]Two [speed=2]Three")).play(
        output_wavfile=str(output), on_audio_chunk=received.append)
    assert [value.strip() for value in stream.engine.spoken] == ["One", "Two", "Three"]
    segments = [s for s in stream.tempo_timeline.segments if s.closed]
    assert [s.speed for s in segments] == [1.0, 0.5, 2.0]
    assert [s.source_duration for s in segments] == pytest.approx([1.0] * 3)
    assert [s.output_duration for s in segments] == pytest.approx([1.0, 2.0, 0.5], abs=0.12)
    with wave.open(str(output), "rb") as wav:
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(wav.getnframes()) == b"".join(received)


def test_new_play_does_not_restore_a_previous_play_scope(stream_factory):
    stream = stream_factory(speed_tags=True)
    stream.feed("One <sr=.5>Two").play()
    stream.speed = 1.25
    stream.feed("</speed>Three <speed=2>Four[/sr]Five").play()
    assert [s.speed for s in stream.tempo_timeline.segments if s.closed] == [1.25, 2.0, 1.25]
    assert [text.strip() for text in stream.engine.spoken[-3:]] == ["Three", "Four", "Five"]


def test_default_speed_and_inline_state_reset_on_next_play(stream_factory):
    stream = stream_factory(speed=1.25, speed_tags=True)
    stream.feed("One [speed=2]Two").play()
    stream.speed = 0.5
    stream.feed("Three [speed=reset]Four").play()
    segments = [s for s in stream.tempo_timeline.segments if s.closed]
    assert [s.speed for s in segments] == [0.5, 0.5]
    assert stream.tempo_timeline.source_duration == pytest.approx(2.0)
    assert stream.tempo_timeline.output_duration == pytest.approx(4.0, abs=0.2)


def test_generator_consuming_engine_handles_controls_without_speaking_them(stream_factory):
    engine = RecordingEngine()
    engine.can_consume_generators = True
    stream = stream_factory(engine, speed_tags=True)
    stream.feed(iter("One [speed=2]Two [speed=reset]Three")).play()
    assert engine.spoken == ["One ", "Two ", "Three"]
    assert [s.speed for s in stream.tempo_timeline.segments if s.closed] == [1.0, 2.0, 1.0]


def test_audio_is_delivered_before_character_source_finishes(stream_factory):
    played, release = threading.Event(), threading.Event()

    def source():
        yield "First [speed=0.5]"
        assert release.wait(3)
        yield "Second"

    stream = stream_factory(speed_tags=True)
    try:
        stream.feed(source()).play_async(on_audio_chunk=lambda _: played.set())
        assert played.wait(2), "The lexer buffered the whole stream"
    finally:
        release.set()
        if stream.play_thread:
            stream.play_thread.join(3)
    assert not stream.is_playing()
    assert [value.strip() for value in stream.engine.spoken] == ["First", "Second"]


def test_inserted_pause_obeys_active_speed(stream_factory):
    stream = stream_factory(speed=2.0, speed_tags=True)
    stream.add_pause("pause", 0.4)
    stream.feed("One[pause]Two").play()
    assert stream.tempo_timeline.source_duration == pytest.approx(2.4)
    assert stream.tempo_timeline.output_duration == pytest.approx(1.2, abs=0.08)


@pytest.mark.parametrize("source", ["One [speed=0.5]Two", "One <sr=.5>Two[/speed]"])
def test_alignment_omits_control_tags_and_returns_derived_output_times(stream_factory, source):
    words = []
    aligner = FakeWordAligner([WordAlignment("word", 0.0, 0.2)])
    stream = stream_factory(speed_tags=True, align_words=True, word_aligner=aligner,
                            alignment_blocking=True, on_word=words.append)
    try:
        stream.feed(source).play()
        assert [call[0].strip() for call in aligner.calls] == ["One", "Two"]
        assert [span.start_time for span in words] == pytest.approx([0.0, 1.0], abs=0.02)
        assert words[-1].end_time == pytest.approx(1.4, abs=0.04)
        assert words[-1].source.startswith("retimed_")
        assert "speed" not in stream.alignment.snapshot().text
        assert stream.alignment.snapshot().duration == pytest.approx(2.0)
        assert stream.tempo_timeline.output_duration == pytest.approx(3.0, abs=0.12)
    finally:
        stream.close()


def test_native_word_times_are_mapped_once_to_their_own_speed_segments(stream_factory):
    class NativeEngine(RecordingEngine):
        def synthesize(self, text, sentence_count=0):
            start = self.audio_duration
            success = super().synthesize(text, sentence_count)
            self.timings.put(TimingInfo(start, start + 0.2, text.strip()))
            self.audio_duration += 1.0
            return success

    words = []
    stream = stream_factory(NativeEngine(), speed_tags=True, on_word=words.append)
    stream.feed("One [speed=0.5]Two [speed=2]Three").play()
    assert [word.word for word in words] == ["One", "Two", "Three"]
    boundaries = stream.tempo_timeline.segments
    third_start = boundaries[2].output_start
    assert [word.start_time for word in words] == pytest.approx([0.0, 1.0, third_start], abs=0.03)
    assert words[-1].end_time == pytest.approx(third_start + 0.1, abs=0.03)
    assert words[-1].source_start_time == pytest.approx(2.0)


def test_speed_markers_do_not_enter_alignment_audio_tap(stream_factory):
    aligner = FakeWordAligner([WordAlignment("word", 0, 0.1)])
    stream = stream_factory(speed_tags=True, align_words=True, word_aligner=aligner)
    try:
        stream.feed("One[speed=2]Two").play()
        assert len(aligner.calls) == 2
        assert stream.player.playback_error is None
    finally:
        stream.close()


def test_speed_processing_error_is_reported_to_sync_caller(stream_factory, monkeypatch):
    stream = stream_factory(speed_tags=True)
    class BrokenProcessor:
        def __init__(self, *args):
            raise ImportError("Install realtimetts[speed]")
    monkeypatch.setattr(text_module, "TempoProcessor", BrokenProcessor)
    with pytest.raises(ImportError, match="realtimetts"):
        stream.feed("One[speed=2]Two").play()
    assert not stream.is_playing()


@pytest.fixture
def mp3_payload():
    av = pytest.importorskip("av")
    output = io.BytesIO()
    with av.open(output, "w", format="mp3") as container:
        audio = container.add_stream("libmp3lame", rate=16000)
        audio.layout = "mono"
        signal = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(signal.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    return output.getvalue()


def compressed_engine(payload):
    class MP3Engine(RecordingEngine):
        def get_stream_info(self):
            return pyaudio.paCustomFormat, -1, -1

        def synthesize(self, text, sentence_count=0):
            self.spoken.append(text)
            for cursor in range(0, len(payload), 73):
                self.queue.put(payload[cursor:cursor + 73])
            return True
    return MP3Engine()


def test_compressed_engine_decodes_once_and_exports_consistent_pcm(stream_factory, mp3_payload, tmp_path):
    from RealtimeTTS.audio_tempo import DecodedAudioQueue
    engine = compressed_engine(mp3_payload)
    stream = stream_factory(engine, speed_tags=True, muted=True)
    decoder = engine.queue
    assert isinstance(decoder, DecodedAudioQueue)
    stream.load_engine(engine)
    assert engine.queue is decoder
    assert not isinstance(decoder.wrapped, DecodedAudioQueue)
    output = tmp_path / "decoded.wav"
    chunks = []
    try:
        stream.feed("One[speed=2]Two").play(output_wavfile=str(output), on_audio_chunk=chunks.append)
        assert stream.player.playback_error is None
        assert len(engine.spoken) == 2
        with wave.open(str(output), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getsampwidth() == 2
            pcm = wav.readframes(wav.getnframes())
        assert pcm == b"".join(chunks)
        values = np.frombuffer(pcm, dtype=np.int16)
        assert np.max(np.abs(values.astype(np.int32))) > 1000
        target = sum(segment.source_duration / segment.speed for segment in stream.tempo_timeline.segments)
        assert stream.tempo_timeline.output_duration == pytest.approx(target, abs=0.1)
    finally:
        stream.close()


def test_changing_default_speed_before_play_enables_compressed_processing(stream_factory, mp3_payload):
    engine = compressed_engine(mp3_payload)
    stream = stream_factory(engine, muted=True)
    stream.speed = 2
    chunks = []
    try:
        stream.feed("One").play(on_audio_chunk=chunks.append)
        assert stream._output_stream_info()[0] == pyaudio.paFloat32
        assert stream.tempo_timeline.output_duration == pytest.approx(0.5, abs=0.08)
        assert chunks
        stream.speed = 1
        stream.load_engine(engine)
        assert stream._output_stream_info()[0] == pyaudio.paFloat32
        stream.feed("Again").play()
        assert stream.tempo_timeline.output_duration == pytest.approx(stream.tempo_timeline.source_duration)
    finally:
        stream.close()


def test_legacy_compressed_alignment_captures_decoded_pcm(stream_factory, mp3_payload):
    aligner = FakeWordAligner([WordAlignment("word", 0.0, 0.2)])
    engine = compressed_engine(mp3_payload)
    stream = stream_factory(engine, speed_tags=True, align_words=True, alignment=False,
                            word_aligner=aligner, alignment_blocking=True, muted=True)
    try:
        stream.feed("One[speed=2]Two").play()
        assert len(aligner.calls) == 2
        for text, frames, rate in aligner.calls:
            assert text in {"One", "Two"}
            assert rate == 24000
            assert 24000 <= frames < 30000
        assert stream.player.playback_error is None
    finally:
        stream.close()


def test_decoder_cancel_fences_late_audio_and_finish(mp3_payload):
    import queue
    from RealtimeTTS.audio_tempo import DecodedAudioQueue
    q = queue.Queue()
    decoded = DecodedAudioQueue(q, 24000)
    decoded.put(mp3_payload[:20])
    decoded.cancel()
    decoded.put(mp3_payload[20:])
    decoded.finish()
    assert q.empty()
    decoded.begin()
    decoded.put(mp3_payload)
    decoded.finish()
    assert not q.empty()


def test_stop_discards_tempo_state_and_next_play_starts_cleanly(stream_factory):
    played = threading.Event()
    stream = stream_factory(speed=0.5, speed_tags=True)
    stream.feed("One[speed=2]Two").play_async(on_audio_chunk=lambda _: played.set())
    assert played.wait(2)
    stream.stop()
    assert not stream.is_playing()
    assert stream.player.timings_list == []
    assert stream.player._tempo_input_seconds == 0
    assert stream.player._tempo_output_seconds == 0
    assert stream.player._tempo_processor is None
    stream.speed = 1
    stream.feed("Fresh").play()
    assert stream.tempo_timeline.source_duration == pytest.approx(1.0)
    assert stream.tempo_timeline.output_duration == pytest.approx(1.0)


def test_speed_one_keeps_other_pcm_formats_without_importing_av(stream_factory, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def reject_av(name, *args, **kwargs):
        if name == "av" or name.startswith("av."):
            raise AssertionError("default playback imported av")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_av)
    monkeypatch.setattr(_FakePyAudio, "get_sample_size",
                        lambda self, fmt: 4 if fmt == pyaudio.paInt32 else 2)
    payload = np.arange(1600, dtype=np.int32).tobytes()
    class Int32Engine(RecordingEngine):
        def get_stream_info(self):
            return pyaudio.paInt32, 1, 16000
        def synthesize(self, text, sentence_count=0):
            self.queue.put(payload)
            return True
    stream = stream_factory(Int32Engine())
    chunks = []
    stream.feed("Default").play(on_audio_chunk=chunks.append)
    assert b"".join(chunks) == payload


def test_marker_buffer_accounting_uses_audio_frames_only():
    import queue
    from RealtimeTTS.stream_player import AudioBufferManager, AudioConfiguration
    q = queue.Queue()
    q.put(SpeedChange(2))
    q.put(b"\0" * 320)
    manager = AudioBufferManager(q, queue.Queue(), AudioConfiguration(rate=16000))
    assert manager.get_from_buffer()[1] == SpeedChange(2)
    assert manager.get_from_buffer()[1] == b"\0" * 320
