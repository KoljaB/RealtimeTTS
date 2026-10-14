"""
TextToAudioStream Module
------------------------
Converts text into real-time audio using one or more TTS engines. Key features include:

- Synthesis: Transforms text or text streams into audio.
- Multi-Engine: Supports engine switching if one fails.
- Async Playback: Handles play, pause, resume, and stop with separate threads.
- Callbacks: Offers hooks for stream events, per-character, and per-word processing.
- Buffer Management: Generates audio chunks based on buffered duration.
- Output Options: Plays audio live or writes to a WAV file.
"""


from .threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator
from .stream_player import StreamPlayer, AudioConfiguration
from .audio_tempo import TempoProcessor, SpeedChange, validate_speed, DecodedAudioQueue
from .tempo_timeline import TempoTimeline
from .inline_markup import InlineMarkupParser
from itertools import groupby
from dataclasses import replace
from typing import Union, Iterator, List, Optional
from .alignment.audio import is_alignable_stream_format
from .alignment.capture import CapturedAudioSegment, CapturingAudioQueue
from .engines import BaseEngine, TimingInfo
from ._audio_backend import pa, pyaudio
import re
import numpy as np
import threading
import traceback
import logging
import queue
import time
import wave

_s2s = None


def _get_stream2sentence():
    """Import stream2sentence only when sentence splitting is actually used."""
    global _s2s
    if _s2s is None:
        import stream2sentence as s2s

        _s2s = s2s
    return _s2s


class TextToAudioStream:
    def __init__(
        self,
        engine: Union[BaseEngine, List[BaseEngine]],
        log_characters: bool = False,
        on_text_stream_start=None,
        on_text_stream_stop=None,
        on_audio_stream_start=None,
        on_audio_stream_stop=None,
        on_character=None,
        on_word=None,
        output_device_index=None,
        mpv_audio_device=None,
        tokenizer: str = "nltk+rule-based",
        language: str = "en",
        muted: bool = False,
        frames_per_buffer: int = pa.paFramesPerBufferUnspecified,
        playout_chunk_size: int = -1,
        level=logging.WARNING,
        align_words: bool = False,
        word_aligner=None,
        alignment_blocking: bool = False,
        alignment=None,
        on_alignment=None,
        on_character_spoken=None,
        speed: float = 1.0,
        speed_tags: bool = False,
        language_tags: bool = False,
        language_voices=None,
    ):
        """
        Initializes the TextToAudioStream.

        Args:
            engine (Union[BaseEngine, List[BaseEngine]]):
                The engine or list of engines used for text-to-audio synthesis.
                `BaseEngine` is the interface or base class defining how
                synthesis is performed. Multiple engines can be provided for
                fallbacks.
                
            log_characters (bool, optional):
                Whether to log each character being processed for synthesis.
                Useful for debugging or monitoring character-level processing.
                Defaults to False.
                
            on_text_stream_start (callable, optional):
                A callback function triggered when the text stream begins.
                This can be used to perform setup actions or display status
                updates.

            on_text_stream_stop (callable, optional):
                A callback function triggered when the text stream ends.
                This can be used to clean up resources or indicate that the
                text-to-speech process has completed.

            on_audio_stream_start (callable, optional):
                A callback function triggered when the audio playback begins.
                Useful for tracking the start of the audio output.

            on_audio_stream_stop (callable, optional):
                A callback function triggered when the audio playback stops.
                Useful for cleaning up or updating the UI once playback ends.

            on_character (callable, optional):
                A callback function triggered for every individual character
                processed during synthesis. This can be useful for real-time
                updates, such as visualizing which character is being processed
                or sent for synthesis.

            on_word (callable, optional):
                A callback function triggered when a word is spoken. This can be
                useful for tracking word-level progress or highlighting spoken
                words in a text display.
                Native timings are used when an engine provides them. For other
                engines, set align_words=True to enable optional forced
                alignment.

            output_device_index (int, optional):
                The index of the audio output device to use for playback.
                If None, the system's default audio output device will be used.
                This index corresponds to the device indices returned by the
                PyAudio interface.

            mpv_audio_device (str, optional):
                The name of the audio device to use for playback.
                If None, the system's default audio output device will be used.
                This name corresponds to the device names returned by the
                MPV interface.
                Use `mpv --audio-device=help` to get the device names.

            tokenizer (str, optional):
                Specifies the tokenizer used to split input text into sentences
                or smaller chunks for synthesis. Supported options are:
                - "nltk+rule-based": Combines NLTK with stream2sentence's
                  local boundary rules. This is the default.
                - "rule-based": Uses only stream2sentence's built-in tokenizer.
                - "nltk": Uses only the Natural Language Toolkit tokenizer.
                - "stanza": Uses the optional Stanza tokenizer.
                Defaults to "nltk+rule-based".
                
            language (str, optional):
                Language code (e.g., "en" for English, "de" for German) used for
                sentence splitting and processing. Ensure the tokenizer
                supports the specified language. Defaults to "en".

            muted (bool, optional):
                If True, disables audio playback on local speakers, allowing
                audio data to be processed or saved to a file without being
                played. This is useful for silent synthesis scenarios or batch
                processing. Defaults to False.

            frames_per_buffer (int, optional):
                Determines how many audio frames PyAudio processes in each
                buffer. If set to `pa.paFramesPerBufferUnspecified`, PyAudio
                chooses an appropriate default value. Lower values may reduce
                latency but increase CPU usage. Higher values may reduce CPU
                load but increase latency. Defaults to PyAudio’s unspecified
                setting.

            playout_chunk_size (int, optional):
                The size (in bytes) of audio chunks played to the output stream
                at a time. If set to -1, the chunk size is determined based on
                `frames_per_buffer` or a default internal value. Smaller chunks
                allow for lower latency but require more frequent processing,
                while larger chunks may introduce latency but reduce overhead.
                Defaults to -1.

            level (int, optional):
                The logging level to use for internal logging. Accepts standard
                Python logging levels, such as `logging.DEBUG`, `logging.INFO`,
                `logging.WARNING`, etc. Defaults to `logging.WARNING`.

            align_words (bool, optional):
                If True, capture synthesized sentence audio and use a forced
                aligner for engines without native word timings.

            word_aligner (optional):
                Custom aligner implementing align_words(text, audio, sample_rate).
                If omitted while align_words is enabled, the omniASR CTC aligner
                is created lazily.

            alignment_blocking (bool, optional):
                Legacy shorthand for bounded synced alignment playback.

            alignment (optional):
                None/off, auto, a mapping or AlignmentConfig. Native timestamps
                are preferred; configured model alignment works across engines.

            on_alignment (callable, optional):
                Receives immutable, coalesced AlignmentSnapshot updates.

            on_character_spoken (callable, optional):
                Receives measured character spans at their playback positions.

            speed (float, optional):
                Initial pitch-preserving output tempo, from 0.25 to 4.0.
                Non-default speeds require realtimetts[speed].

            speed_tags (bool, optional):
                Enable incremental speed controls and scopes in bracket or
                XML-style notation, e.g. <speed factor="0.85">...</speed>.
                Configured voice tag delimiters also work. Defaults to False.

            language_tags (bool, optional):
                Enable scoped language controls such as <es>...</es>,
                <es slow>...</es> and <es fast>...</es>. Requires an engine
                implementing set_language. Also enables inline speed controls.

            language_voices (mapping, optional):
                Optional ISO language-code to engine voice mapping. Language
                scopes restore the previous voice, language and speed.
        """
        self.log_characters = log_characters
        self.on_text_stream_start = on_text_stream_start
        self.on_text_stream_stop = on_text_stream_stop
        self.on_audio_stream_start = on_audio_stream_start
        self.on_audio_stream_stop = on_audio_stream_stop
        self.output_device_index = output_device_index
        self.mpv_audio_device = mpv_audio_device
        self.on_word_spoken = on_word
        self.output_wavfile = None
        self.chunk_callback = None
        self.wf = None
        self.abort_events = []
        self.tokenizer = tokenizer
        self.language = language
        self.global_muted = muted
        self._speed = validate_speed(speed)
        self.language_tags = bool(language_tags)
        self.language_voices = dict(language_voices or {})
        self.speed_tags = bool(speed_tags or language_tags)
        self._play_default_language = language
        self._synthesis_language = language
        self._language_baselines = {}
        self._inline_voice_baseline = None
        self._language_error = None
        self._synthesis_speed = self._speed
        self._play_default_speed = self._speed
        self._inline_markup_parser = None
        self.tempo_timeline = TempoTimeline(self._speed)
        self._player_output_offset = 0.0
        self._decoded_audio_queue = None
        self._speed_error = None
        self._on_alignment_callback = on_alignment
        self._on_character_spoken_callback = on_character_spoken
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size
        self.align_words = bool(align_words)
        self.word_aligner = word_aligner
        self.alignment_blocking = bool(alignment_blocking)
        self._alignment_capture_queue = None
        self._alignment_lock = threading.Lock()
        self._alignment_threads = []
        self._alignment_generation = 0
        from .alignment.timeline import AlignmentConfig
        from .alignment.controller import AlignmentController
        if alignment is None and align_words:
            alignment = {"mode": "synced" if alignment_blocking else "live"}
        alignment_config = AlignmentConfig.coerce(
            alignment, aligner=word_aligner, language=language
        )
        self.alignment = (
            AlignmentController(alignment_config, on_update=self._on_alignment_update,
                                on_word=self._on_aligned_word,
                                on_character=self._on_aligned_character)
            if alignment_config is not None else None
        )
        self._alignment_binding = None
        self._alignment_player_offset = 0.0
        self.player = None
        self.play_lock = threading.Lock()
        self.is_playing_flag = False
        self._volume = 1.0  # Default volume at 100%
        self.voice_switch_tags = {}
        self.pause_tags = {}
        self.voice_tag_start = "["
        self.voice_tag_end = "]"
        self.active_voice_tag = None
        self.active_voice = None

        self._create_iterators()

        # Initialize the play_thread attribute
        # (used for playing audio in a separate thread)
        self.play_thread = None

        # Initialize an attribute to store generated text
        self.generated_text = ""

        # A flag to indicate if the audio stream is currently running or not
        self.stream_running = False

        self.on_character = on_character

        self.engine_index = 0
        if isinstance(engine, list):
            # Handle the case where engine is a list of BaseEngine instances
            self.engines = engine
        else:
            # Handle the case where engine is a single BaseEngine instance
            self.engines = [engine]

        self.load_engine(self.engines[self.engine_index])

    def load_engine(self, engine: BaseEngine):
        """
        Loads the synthesis engine and prepares the audio player for stream playback.
        This method sets up the engine that will be used for text-to-audio conversion, extracts the necessary stream information like format, channels, and rate from the engine, and initializes the StreamPlayer if the engine does not support consuming generators directly.

        Args:
            engine (BaseEngine): The synthesis engine to be used for converting text to audio.
        """

        # Store the engine instance (responsible for text-to-audio conversion)
        self.engine = engine

        # Extract stream information (format, channels, rate) from the engine
        format, channels, rate = self.engine.get_stream_info()
        self._decoded_audio_queue = None
        if self.alignment is not None:
            self._alignment_binding = self.alignment.bind(self.engine)
            format, channels, rate = self._alignment_binding.output_format
            self._alignment_player_offset = self.alignment.snapshot().duration
        elif format == pyaudio.paCustomFormat and (self.speed_tags or self._speed != 1.0 or isinstance(self.engine.queue, DecodedAudioQueue)):
            # Decode once before both tempo processing and output callbacks.
            decoder_queue = self.engine.queue if isinstance(self.engine.queue, DecodedAudioQueue) else DecodedAudioQueue(self.engine.queue, rate if rate > 0 else 24000)
            self.engine.queue = decoder_queue
            self._decoded_audio_queue = decoder_queue
            format, channels, rate = decoder_queue.output_format

        self._alignment_capture_queue = None
        if self.alignment is None and self._should_capture_alignment_audio(format, channels, rate):
            capture_input = self._decoded_audio_queue.wrapped if self._decoded_audio_queue else self.engine.queue
            if isinstance(capture_input, CapturingAudioQueue):
                self._alignment_capture_queue = capture_input
            else:
                self._alignment_capture_queue = CapturingAudioQueue(capture_input, format, channels, rate)
            if self._decoded_audio_queue is not None:
                # Capture canonical PCM, rather than the incoming MP3 packets.
                self._decoded_audio_queue.wrapped = self._alignment_capture_queue
            else:
                self.engine.queue = self._alignment_capture_queue

        # An engine can opt in when stream2sentence's one-time import would
        # otherwise dominate its first-response latency.
        if (
            not self.engine.can_consume_generators
            and self.engine.preload_sentence_tokenizer
        ):
            _get_stream2sentence()

        # Check if the engine doesn't support consuming generators directly
        config = AudioConfiguration(
            format,
            channels,
            rate,
            self.output_device_index,
            self.mpv_audio_device,
            muted=self.global_muted,
            frames_per_buffer=self.frames_per_buffer,
            playout_chunk_size=self.playout_chunk_size,
        )

        playback_queue = self.engine.queue
        if self._alignment_binding is not None:
            playback_queue = self._alignment_binding.original_queue
        elif self._decoded_audio_queue is not None:
            playback_queue = self._decoded_audio_queue.wrapped
        if isinstance(playback_queue, CapturingAudioQueue):
            playback_queue = playback_queue._queue
        self._player_output_offset = self.tempo_timeline.output_duration
        self.player = StreamPlayer(
            playback_queue,
            self.engine.timings,
            config,
            on_playback_start=self._on_audio_stream_start,
            on_word_spoken=None if self.alignment is not None else self._on_word_spoken,
            speed=self._synthesis_speed,
            tempo_timeline=self.tempo_timeline,
        )
        if self.alignment is not None:
            self.player.on_audio_progress = self._on_alignment_progress
            self.player.before_audio_chunk = self._before_alignment_audio
        
        # Set initial volume if player was created
        if hasattr(self, '_volume'):
            self.player.set_volume(self._volume)

        logging.info(f"loaded engine {self.engine.engine_name}")

    @property
    def speed(self) -> float:
        """Default speed for the next play operation; inline tags override it."""
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = validate_speed(value)

    @property
    def volume(self) -> float:
        """Get the current volume (0.0 to 1.0)"""
        return self._volume

    @volume.setter
    def volume(self, value: float):
        """Set the volume (0.0 to 1.0)"""
        self.set_volume(value)

    def set_volume(self, volume: float):
        """
        Set the playback volume.
        
        Args:
            volume (float): Volume level from 0.0 (muted) to 1.0 (full volume)
        """
        self._volume = max(0.0, min(1.0, volume))
        if self.player:
            self.player.set_volume(self._volume)

    def feed(self, text_or_iterator: Union[str, Iterator[str]]):
        """
        Feeds text or an iterator to the stream.

        Args:
            text_or_iterator: Text or iterator to be fed.

        Returns:
            Self instance.
        """
        self.char_iter.add(text_or_iterator)
        return self

    def add_voice(self, tag: str, voice):
        """
        Registers a voice switch tag.

        Args:
            tag (str): Tag content without delimiters, for example "angry".
            voice: Voice object or identifier accepted by the current engine.

        Returns:
            Self instance.
        """
        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ValueError("Voice tag must not be empty.")

        self.voice_switch_tags[normalized_tag] = voice
        return self

    def add_emotion(self, tag: str, voice):
        """
        Backward-compatible alias for add_voice().
        """
        return self.add_voice(tag, voice)

    def remove_voice(self, tag: str):
        """
        Removes a registered voice switch tag.

        Args:
            tag (str): Tag content without delimiters.

        Returns:
            Self instance.
        """
        self.voice_switch_tags.pop(tag.strip(), None)
        return self

    def remove_emotion(self, tag: str):
        """
        Backward-compatible alias for remove_voice().
        """
        return self.remove_voice(tag)

    def clear_voices(self):
        """
        Clears all registered voice switch tags.

        Returns:
            Self instance.
        """
        self.voice_switch_tags.clear()
        return self

    def add_pause(self, tag: str = "pause", duration: float = 0.5):
        """
        Registers a pause tag with a default silence duration in seconds.

        Args:
            tag (str): Tag content without delimiters, for example "pause".
            duration (float): Pause duration in seconds.

        Returns:
            Self instance.
        """
        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ValueError("Pause tag must not be empty.")
        if duration < 0:
            raise ValueError("Pause duration must not be negative.")

        self.pause_tags[normalized_tag] = float(duration)
        return self

    def remove_pause(self, tag: str):
        """
        Removes a registered pause tag.

        Args:
            tag (str): Tag content without delimiters.

        Returns:
            Self instance.
        """
        self.pause_tags.pop(tag.strip(), None)
        return self

    def clear_pauses(self):
        """
        Clears all registered pause tags.

        Returns:
            Self instance.
        """
        self.pause_tags.clear()
        return self

    def set_voice_tag_delimiters(self, start: str = "[", end: str = "]"):
        """
        Configures the delimiters used to detect voice switch tags.

        Args:
            start (str): Opening delimiter, for example "[" or "**".
            end (str): Closing delimiter, for example "]" or "**".

        Returns:
            Self instance.
        """
        if not start:
            raise ValueError("Voice tag start delimiter must not be empty.")
        if not end:
            raise ValueError("Voice tag end delimiter must not be empty.")

        self.voice_tag_start = start
        self.voice_tag_end = end
        return self

    def play_async(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = True,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "",
        tokenize_sentences=None,
        language: str = "",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        comma_silence_duration=0.0,
        sentence_silence_duration=0.0,
        default_silence_duration=0.0,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。—",
        force_first_fragment_after_words=30,
        debug=False,
        fragment_lookahead_words: int = 0,
    ):
        """
        Async handling of text to audio synthesis, see play() method.
        """
        if not self.is_playing_flag:
            self.is_playing_flag = True
            args = (
                fast_sentence_fragment,
                fast_sentence_fragment_allsentences,
                fast_sentence_fragment_allsentences_multiple,
                buffer_threshold_seconds,
                minimum_sentence_length,
                minimum_first_fragment_length,
                log_synthesized_text,
                reset_generated_text,
                output_wavfile,
                on_sentence_synthesized,
                before_sentence_synthesized,
                on_audio_chunk,
                tokenizer,
                tokenize_sentences,
                language,
                context_size,
                context_size_look_overhead,
                comma_silence_duration,
                sentence_silence_duration,
                default_silence_duration,
                muted,
                sentence_fragment_delimiters,
                force_first_fragment_after_words,
                True,
                debug,
                fragment_lookahead_words,
            )
            self.play_thread = threading.Thread(target=self.play, args=args)
            self.play_thread.start()
        else:
            logging.warning("play_async() called while already playing audio, skipping")

    def play(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = False,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "",
        tokenize_sentences=None,
        language: str = "en",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        comma_silence_duration=0.0,
        sentence_silence_duration=0.0,
        default_silence_duration=0.0,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。—",
        force_first_fragment_after_words=30,
        is_external_call=True,
        debug=False,
        fragment_lookahead_words: int = 0,
    ):
        """
        Handles the synthesis of text to audio.
        Plays the audio stream and waits until it is finished playing.
        If the engine can't consume generators, it utilizes a player.

        Args:
        - fast_sentence_fragment: Determines if sentence fragments should be quickly yielded. Useful when a faster response is desired even if a sentence isn't complete.
        - fast_sentence_fragment_allsentences: Fast_sentence_fragment only works on the first sentence. Set this to True if you want to work it on every sentence.
        - fast_sentence_fragment_allsentences_multiple: Can yield multiple sentence fragments, not only a single one.
        - buffer_threshold_seconds (float): Time in seconds for the buffering threshold, influencing the flow and continuity of audio playback. Set to 0 to deactivate. Default is 0.
          - How it Works: The system verifies whether there is more audio content in the buffer than the duration defined by buffer_threshold_seconds. If so, it proceeds to synthesize the next sentence, capitalizing on the remaining audio to maintain smooth delivery. A higher value means more audio is pre-buffered, which minimizes pauses during playback. Adjust this upwards if you encounter interruptions.
          - Helps to decide when to generate more audio based on buffered content.
        - minimum_sentence_length (int): The minimum number of characters a sentence must have. If a sentence is shorter, it will be concatenated with the following one, improving the overall readability. This parameter does not apply to the first sentence fragment, which is governed by `minimum_first_fragment_length`. Default is 10 characters.
        - minimum_first_fragment_length (int): The minimum number of characters required for the first sentence fragment before yielding. Default is 10 characters.
        - log_synthesized_text: If True, logs the synthesized text chunks.
        - reset_generated_text: If True, resets the generated text.
        - output_wavfile: If set, saves the audio to the specified WAV file.
        - on_sentence_synthesized: Callback function that gets called after hen a single sentence fragment was synthesized.
        - before_sentence_synthesized: Callback function that gets called before a single sentence fragment gets synthesized.
        - on_audio_chunk: Callback function that gets called when a single audio chunk is ready.
        - tokenizer: Tokenizer to use for sentence splitting. Leave empty to use the stream setting ("nltk+rule-based" by default). Stanza remains optional and requires the `RealtimeTTS[stanza]` extra.
        - tokenize_sentences (Callable): A function that tokenizes sentences from the input text. You can write your own lightweight tokenizer here if you are unhappy with nltk and stanza. Defaults to None. Takes text as string and should return splitted sentences as list of strings.
        - language: Language to use for sentence splitting.
        - context_size: The number of characters used to establish context for sentence boundary detection. A larger context improves the accuracy of detecting sentence boundaries. Default is 12 characters.
        - context_size_look_overhead: The number of characters to look ahead when determining sentence boundaries. This helps in identifying the end of a sentence more accurately. Default is 12 characters.
        - comma_silence_duration: The duration of silence to insert after a comma in seconds. Default is 0.0 seconds.
        - sentence_silence_duration: The duration of silence to insert after a sentence in seconds. Default is 0.0 seconds.
        - default_silence_duration: The default duration of silence to insert after a sentence in seconds. Default is 0.0 seconds.
        - muted: If True, disables audio playback via local speakers (in case you want to synthesize to file or process audio chunks). Default is False.
        - sentence_fragment_delimiters (str): A string of characters that are
            considered sentence delimiters. Default is ".?!;:,\n…)]}。-".
        - force_first_fragment_after_words (int): The number of words after
            which the first sentence fragment is forced to be yielded.
            Default is 30 words.
        - is_external_call: If True, the method is called from an external source.
        - debug: If True, enables debug mode.
        """
        if self.global_muted:
            muted = True

        if is_external_call:
            if not self.play_lock.acquire(blocking=False):
                logging.warning("play() called while already playing audio, skipping")
                return
            self._alignment_generation += 1
            try:
                if (self.engine.get_stream_info()[0] == pyaudio.paCustomFormat
                        and self._decoded_audio_queue is None and self.alignment is None
                        and (self.speed_tags or self._speed != 1.0)):
                    self.load_engine(self.engine)
            except Exception:
                self.play_lock.release()
                raise
            self._language_baselines = {}
            self._inline_voice_baseline = (self.active_voice_tag, self.active_voice)
            self._language_error = None
            if self.language_tags:
                try:
                    self._remember_inline_language_engine()
                    self._play_default_language = self.engine.get_language()
                    self._synthesis_language = self._play_default_language
                except Exception:
                    self.play_lock.release()
                    raise
            self._synthesis_speed = self._speed
            self._play_default_speed = self._speed
            self._inline_markup_parser = None
            self._speed_error = None
            self.tempo_timeline.reset(self._speed)
            self._player_output_offset = 0.0
            self.player.speed = self._speed
            self.player.seconds_played = 0.0
            self.engine.reset_audio_duration()
            if self.alignment is not None:
                self.alignment.reset()
                self._alignment_player_offset = 0.0
                self.player.seconds_played = 0.0
            if self._alignment_capture_queue:
                self._alignment_capture_queue.reset_tracking()

        self.is_playing_flag = True
        self.error_flag = False

        # Log the start of the stream
        logging.info("stream start")

        tokenizer = tokenizer if tokenizer else self.tokenizer
        language = language if language else self.language

        # Set the stream_running flag to indicate the stream is active
        self.stream_start_time = time.time()
        self.stream_running = True
        abort_event = threading.Event()
        self.abort_events.append(abort_event)

        if self.player:
            self.player.mute(muted)
        elif hasattr(self.engine, "set_muted"):
            self.engine.set_muted(muted)

        self.output_wavfile = output_wavfile
        self.chunk_callback = on_audio_chunk

        if output_wavfile:
            if self._is_engine_mpeg():
                self.wf = open(output_wavfile, "wb")
            else:
                self.wf = wave.open(output_wavfile, "wb")
                _, channels, rate = self._output_stream_info()
                self.wf.setnchannels(channels)
                self.wf.setsampwidth(2)
                self.wf.setframerate(rate)

        # Initialize the generated_text variable
        if reset_generated_text:
            self.generated_text = ""

        # Check if the engine can handle generators directly
        if self.engine.can_consume_generators:
            try:
                # Start the audio player to handle playback
                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                # Control boundaries end only this synthesis segment, not playback.
                self.char_iter.log_characters = self.log_characters
                while True:
                    source = self.char_iter
                    for is_text, actions in groupby(
                        self._iter_inline_actions(source), lambda item: item[0] == "text"
                    ):
                        if abort_event.is_set():
                            break
                        if not is_text:
                            for action_type, action_value in actions:
                                self._apply_inline_action(action_type, action_value)
                            continue
                        characters = (value for _, value in actions)
                        if self._decoded_audio_queue is not None:
                            self._decoded_audio_queue.begin()
                        if self._alignment_binding is not None:
                            self._alignment_binding.begin("", dynamic=True)
                            success = False
                            try:
                                success = self.engine.synthesize(self._aligned_input(characters)) is not False
                            finally:
                                self._alignment_binding.finish(success=success)
                        else:
                            try:
                                self.engine.synthesize(characters)
                            finally:
                                if self._decoded_audio_queue is not None:
                                    self._finish_decoded_audio()

                    if (not self.language_tags or abort_event.is_set()
                            or self.char_iter is source or not self.char_iter.items
                            or self.char_iter.immediate_stop.is_set()):
                        break
            finally:

                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

                if is_external_call:
                    if self.on_audio_stream_stop:
                        self.on_audio_stream_stop()

                # Once done, set the stream running flag to False and log the stream stop
                logging.info("stream stop")

                # Accumulate the generated text and reset the character iterators
                self.generated_text += self.char_iter.iterated_text

                if not self.language_tags or self.char_iter.immediate_stop.is_set():
                    self._create_iterators()

                if is_external_call:
                    self._restore_inline_language()
                    self._finish_alignment_run()
                    self._alignment_generation += 1
                    self.is_playing_flag = False
                    self.play_lock.release()
        else:
            worker_thread = None
            sentence_queue = queue.Queue()
            try:
                # Start the audio player to handle playback
                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                # Generate sentences from the characters
                s2s = _get_stream2sentence()
                splitter_options = dict(
                    context_size=context_size,
                    context_size_look_overhead=context_size_look_overhead,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    quick_yield_single_sentence_fragment=fast_sentence_fragment,
                    quick_yield_for_all_sentences=fast_sentence_fragment_allsentences,
                    quick_yield_every_fragment=fast_sentence_fragment_allsentences_multiple,
                    cleanup_text_links=True,
                    cleanup_text_emojis=True,
                    tokenize_sentences=tokenize_sentences,
                    tokenizer=tokenizer,
                    language=language,
                    log_characters=self.log_characters,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    fragment_lookahead_words=fragment_lookahead_words,
                    debug=debug,
                )

                # Each control boundary flushes preceding text through the splitter.
                # Ordinary text remains lazy, including character-at-a-time input.
                chunk_generator = self._iter_synthesis_actions(
                    self.thread_safe_char_iter, s2s, splitter_options,
                    buffer_threshold_seconds, log_synthesized_text,
                )

                sentence_count = 0

                def synthesize_worker_loop():
                    nonlocal sentence_count
                    while not abort_event.is_set():
                        sentence_item = sentence_queue.get()
                        if sentence_item is None:  # Sentinel value to stop the worker
                            break

                        action_type, action_value = sentence_item
                        if action_type in ("speed", "language"):
                            try:
                                self._apply_inline_action(action_type, action_value)
                            except Exception as exc:
                                if action_type == "language":
                                    self._language_error = exc
                                else:
                                    self._speed_error = exc
                                abort_event.set()
                            finally:
                                sentence_queue.task_done()
                            continue
                        if action_type == "voice":
                            tag, voice = action_value
                            try:
                                self._set_active_voice(tag, voice)
                            except Exception as e:
                                logging.warning(
                                    f'failed to switch voice for tag "{tag}": {e}'
                                )
                                if self.language_tags:
                                    self._language_error = e
                                    abort_event.set()
                            finally:
                                sentence_queue.task_done()
                            continue
                        if action_type == "pause":
                            tag, duration = action_value
                            try:
                                logging.info(
                                    f"Applying pause tag '{tag}' for {duration:.3f}s"
                                )
                                self._enqueue_silence(duration)
                            finally:
                                sentence_queue.task_done()
                            continue

                        sentence = action_value

                        sentence_count += 1

                        synthesis_successful = False
                        if log_synthesized_text:
                            print(f"\033[96m\033[1m⚡ synthesizing\033[0m \033[37m→ \033[2m'\033[22m{sentence}\033[2m'\033[0m")

                        while not synthesis_successful:
                            try:
                                if abort_event.is_set():
                                    break

                                if before_sentence_synthesized:
                                    before_sentence_synthesized(sentence)

                                if self._alignment_binding is not None:
                                    self._alignment_binding.begin(sentence)
                                    success = False
                                    try:
                                        success = self.engine.synthesize(sentence, sentence_count)
                                    finally:
                                        self._alignment_binding.finish(success=bool(success))
                                else:
                                    capture_started = self._start_alignment_capture(sentence)
                                    if self._decoded_audio_queue is not None:
                                        self._decoded_audio_queue.begin()
                                    try:
                                        success = self.engine.synthesize(sentence, sentence_count)
                                    finally:
                                        if self._decoded_audio_queue is not None:
                                            self._finish_decoded_audio()
                                        alignment_segment = self._finish_alignment_capture(capture_started)
                                    if success and alignment_segment:
                                        self._schedule_alignment(alignment_segment)

                                end_sentence_delimeters = ".!?…。¡¿"
                                mid_sentence_delimeters = ";:,\n()[]{}-“”„”—/|《》"

                                text_stripped = sentence.strip()
                                if text_stripped and text_stripped[-1] in end_sentence_delimeters:
                                    silence_duration = sentence_silence_duration
                                elif text_stripped and text_stripped[-1] in mid_sentence_delimeters:
                                    silence_duration = comma_silence_duration
                                else:
                                    silence_duration = default_silence_duration

                                self._enqueue_silence(silence_duration)


                                if success:
                                    if on_sentence_synthesized:
                                        on_sentence_synthesized(sentence)
                                    synthesis_successful = True
                                else:
                                    logging.warning(
                                        f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}", unknown error'
                                    )

                            except Exception as e:
                                logging.warning(
                                    f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}" with error: {e}'
                                )
                                tb_str = traceback.format_exc()
                                print(f"Traceback: {tb_str}")
                                print(f"Error: {e}")

                            if log_synthesized_text:
                                print(f"\033[92m\033[1m✔ SYNTHESIS FINISHED\033[0m")
                            
                            if not synthesis_successful:
                                if len(self.engines) == 1:
                                    time.sleep(0.2)
                                    logging.warning(
                                        f"engine {self.engine.engine_name} is the only engine available, can't switch to another engine"
                                    )
                                    break
                                else:
                                    logging.warning(
                                        "fallback engine(s) available, switching to next engine"
                                    )
                                    self.engine_index = (self.engine_index + 1) % len(
                                        self.engines
                                    )

                                    self.player.stop()
                                    self.load_engine(self.engines[self.engine_index])
                                    self._reapply_active_voice()
                                    if self.language_tags:
                                        self._apply_inline_action("language", self._synthesis_language)
                                    self.player.start()
                                    self.player.on_audio_chunk = self._on_audio_chunk

                        sentence_queue.task_done()

                def synthesize_worker():
                    if not self.language_tags:
                        return synthesize_worker_loop()
                    try:
                        synthesize_worker_loop()
                    except Exception as exc:
                        self._language_error = exc
                        self.error_flag = True
                        abort_event.set()

                worker_thread = threading.Thread(target=synthesize_worker)
                worker_thread.daemon = True
                worker_thread.start()

                # Iterate through the synthesized chunks and feed them to the engine for audio synthesis
                for action_type, action_value in chunk_generator:
                    if abort_event.is_set():
                        break
                    if self.player.playback_error is not None:
                        abort_event.set()
                        break
                    if action_type == "text":
                        action_value = action_value.strip()
                        if not action_value:
                            continue
                    sentence_queue.put((action_type, action_value))

                # Signal to the worker to stop
                sentence_queue.put(None)
                worker_thread.join()

            except Exception as e:
                self.error_flag = True
                if self.language_tags:
                    self._language_error = e
                    abort_event.set()
                logging.warning(
                    f"error in play() with engine {self.engine.engine_name}: {e}"
                )
                tb_str = traceback.format_exc()
                print(f"Traceback: {tb_str}")
                print(f"Error: {e}")

            finally:
                if worker_thread is not None and worker_thread.is_alive():
                    abort_event.set()
                    sentence_queue.put(None)
                    self.engine.stop()
                    worker_thread.join()
                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

            if (not self.error_flag
                and len(self.char_iter.items) > (0 if self.language_tags else 1)
                and self.char_iter.iterated_text == ""
                and not self.char_iter.immediate_stop.is_set()
                and not abort_event.is_set()
                and not (sentence_count == 0 and not is_external_call)
            ):

                logging.info(f"{len(self.char_iter.items)} unprocessed characters left, recursively calling play()")

                # new text was feeded while playing audio but after the last character was processed
                # we need to start another play() call (!recursively!)
                self.play(
                    fast_sentence_fragment=fast_sentence_fragment,
                    buffer_threshold_seconds=buffer_threshold_seconds,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    log_synthesized_text=log_synthesized_text,
                    reset_generated_text=False,
                    output_wavfile=output_wavfile,
                    on_sentence_synthesized=on_sentence_synthesized,
                    on_audio_chunk=on_audio_chunk,
                    tokenizer=tokenizer,
                    language=language,
                    context_size=context_size,
                    muted=muted,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    fragment_lookahead_words=fragment_lookahead_words,
                    is_external_call=False,
                    debug=debug,
                )

            if is_external_call:
                if self.on_audio_stream_stop:
                    self.on_audio_stream_stop()

                self._restore_inline_language()
                self._finish_alignment_run()
                self._alignment_generation += 1
                self.is_playing_flag = False
                self.play_lock.release()

        failure = self._language_error or self._speed_error or self.player.playback_error
        if failure is not None:
            self.error_flag = True
            raise failure

    def pause(self):
        """
        Pauses playback of the synthesized audio stream (won't work properly with elevenlabs).
        """
        if self.is_playing():
            logging.info("stream pause")
            self.player.pause()

    def resume(self):
        """
        Resumes a previously paused playback of the synthesized audio stream
        - won't work properly with elevenlabs
        """
        if self.is_playing():
            logging.info("stream resume")
            self.player.resume()

    def stop(self):
        """
        Stops the playback of the synthesized audio stream immediately.
        """
        self._alignment_generation += 1
        if self.alignment is not None:
            self.alignment.cancel()

        if self._decoded_audio_queue is not None:
            self._decoded_audio_queue.cancel()
        if self.engine:
            self.engine.stop()

        for abort_event in self.abort_events:
            abort_event.set()

        if self.is_playing():
            self.char_iter.stop()
            self.player.resume()
            self.player.stop(immediate=True)
            self.stream_running = False

        if self.play_thread is not None:
            if self.play_thread.is_alive():
                self.play_thread.join()
            self.play_thread = None

        self._inline_markup_parser = None
        self._create_iterators()

    def text(self):
        """
        Retrieves the text that has been fed into the stream.

        Returns:
            The accumulated text.
        """
        if self.generated_text:
            return self.generated_text
        return self.thread_safe_char_iter.accumulated_text()

    def is_playing(self):
        """
        Checks if the stream is currently playing.

        Returns:
            Boolean indicating if the stream is playing.
        """
        return self.stream_running or self.is_playing_flag

    def _on_audio_stream_start(self):
        """
        Handles the start of the audio stream.

        This method is called when the audio stream starts. It calculates and logs the latency from the stream's start time to the time when the first chunk of audio is received. If a callback for handling the start of the audio stream is set (on_audio_stream_start), it is executed.

        No parameters or returns.
        """
        latency = time.time() - self.stream_start_time
        logging.info(f"Audio stream start, latency to first chunk: {latency:.2f}s")

        if self.on_audio_stream_start:
            self.on_audio_stream_start()

    def _on_word_spoken(self, word):
        """
        Handles the spoken word event.
        """
        if self.on_word_spoken:
            if self.tempo_timeline.changed:
                retimed = TimingInfo(
                    self.tempo_timeline.source_to_output(word.start_time),
                    self.tempo_timeline.source_to_output(word.end_time), word.word,
                )
                retimed.source_start_time, retimed.source_end_time = word.start_time, word.end_time
                retimed.source = "retimed"
                retimed.derived = True
                retimed.quality = "retimed_derived"
                word = retimed
            self.on_word_spoken(word)

    def _retime_span(self, span):
        if not self.tempo_timeline.changed:
            return span
        closed_end = max((segment.source_end for segment in self.tempo_timeline.segments if segment.closed), default=0.0)
        state = span.state if span.end_time <= closed_end else "provisional"
        return replace(span, state=state,
                       start_time=self.tempo_timeline.source_to_output(span.start_time),
                       end_time=self.tempo_timeline.source_to_output(span.end_time),
                       source="retimed_" + span.source)

    def _on_aligned_word(self, span):
        if self.on_word_spoken is not None:
            self.on_word_spoken(self._retime_span(span))

    def _on_aligned_character(self, span):
        if self._on_character_spoken_callback is not None:
            self._on_character_spoken_callback(self._retime_span(span))

    def _on_alignment_update(self, snapshot):
        if self._on_alignment_callback is None:
            return
        if self.tempo_timeline.changed:
            clock_complete = (self.tempo_timeline.current_speed is None
                              and self.tempo_timeline.source_duration >= snapshot.duration - 1e-8)
            snapshot = replace(snapshot, final=snapshot.final and (clock_complete or snapshot.status == "cancelled"),
                               words=tuple(self._retime_span(span) for span in snapshot.words),
                               characters=tuple(self._retime_span(span) for span in snapshot.characters),
                               duration=self.tempo_timeline.source_to_output(snapshot.duration))
        self._on_alignment_callback(snapshot)

    def _on_audio_chunk(self, chunk):
        """
        Postprocessing of single chunks of audio data.
        This method is called for each chunk of audio data processed. It first determines the audio stream format.
        If the format is `pyaudio.paFloat32`, we convert to paInt16.

        Args:
            chunk (bytes): The audio data chunk to be processed.
        """
        format, channels, sample_rate = self._output_stream_info()

        if format == pyaudio.paFloat32:
            audio_data = np.frombuffer(chunk, dtype=np.float32)
            audio_data = np.int16(audio_data * 32767)
            chunk = audio_data.tobytes()

        if self.output_wavfile and self.wf:
            if self._is_engine_mpeg():
                self.wf.write(chunk)
            else:
                if self.wf.getnframes() == 0 and self.player:
                    actual_rate = self.player.audio_stream.actual_sample_rate
                    if actual_rate > 0:
                        self.wf.setframerate(actual_rate)
                self.wf.writeframes(chunk)

        if self.chunk_callback:
            self.chunk_callback(chunk)

    def _on_last_character(self):
        """
        This method is invoked when the last character of the text stream has been processed.
        It logs information and triggers a callback, if defined.
        """

        # If an on_text_stream_stop callback is defined, invoke it to signal the end of the text stream
        if self.on_text_stream_stop:
            self.on_text_stream_stop()

        # If log_characters flag is True, print a new line for better log readability
        if self.log_characters:
            print()

        self._create_iterators()

    def _create_iterators(self):
        """
        Creates iterators required for text-to-audio streaming.

        This method initializes two types of iterators:

        1. `CharIterator`: Responsible for managing individual characters during the streaming process.
        - It takes callbacks for events like when a character is processed (`on_character`), when the first text chunk is encountered (`on_first_text_chunk`), and when the last text chunk is encountered (`on_last_text_chunk`).

        2. `AccumulatingThreadSafeGenerator`: A thread-safe wrapper around `CharIterator`.
        - Ensures that the character iterator can be safely accessed from multiple threads.
        """

        # Create a CharIterator instance for managing individual characters
        self.char_iter = CharIterator(
            on_character=self._on_character,
            on_first_text_chunk=self.on_text_stream_start,
            on_last_text_chunk=self._on_last_character,
        )

        # Create a thread-safe version of the char iterator
        self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)

    def _on_character(self, char: str):
        """
        This method is called for each character that is processed in the text stream.
        It accumulates the characters and invokes a callback.

        Args:
            char (str): The character currently being processed.
        """
        # The control lexer appends only spoken text to the alignment source.
        # If an on_character callback is defined, invoke it for the current character
        if self.on_character:
            self.on_character(char)

        self.generated_text += char

    def _new_inline_markup_parser(self):
        # Engines with public voice state can restore a top-level voice scope.
        # Otherwise a registered point tag (e.g. [normal]) establishes a baseline.
        voice = getattr(self.engine, "current_voice", None)
        if voice is None:
            voice = getattr(self.engine, "voice", None)
        initial_voice = ((None, voice) if voice is not None else
                         (self.active_voice_tag, self.active_voice)
                         if self.active_voice is not None else None)
        return InlineMarkupParser(
            self._play_default_speed,
            speed_tags=self.speed_tags,
            language_tags=self.language_tags,
            default_language=self._play_default_language,
            language_voices=self.language_voices,
            voice_tags=self.voice_switch_tags,
            pause_tags=self.pause_tags,
            initial_voice=initial_voice,
            delimiters=(self.voice_tag_start, self.voice_tag_end),
        )

    def _extract_inline_actions(self, text: str):
        """Use the streaming grammar for full-text extraction as well."""
        actions = []
        for is_text, group in groupby(
            self._new_inline_markup_parser().iter_actions((text,)),
            lambda item: item[0] == "text",
        ):
            if is_text:
                value = "".join(value for _, value in group)
                if value:
                    actions.append(("text", value))
            else:
                actions.extend(group)
        return actions

    def _iter_inline_actions(self, source):
        """Normalize bounded inline markup without buffering its spoken body."""
        if self.is_playing_flag:
            # Late feed() input may continue through an internal play() call.
            # Keep its scope stack until the next external play operation.
            if self._inline_markup_parser is None:
                self._inline_markup_parser = self._new_inline_markup_parser()
            parser = self._inline_markup_parser
        else:
            parser = self._new_inline_markup_parser()
        for kind, value in parser.iter_actions(
            source, stopped=lambda: self.char_iter.immediate_stop.is_set()
        ):
            if kind == "text" and self.alignment is not None:
                self.alignment.append_source(value)
            yield kind, value

    def _iter_synthesis_actions(self, source, splitter, options, threshold, log_chunks):
        language = options["language"]
        if self.language_tags and self._inline_markup_parser is not None:
            code = InlineMarkupParser.language_code(self._inline_markup_parser.current_language)
            language = language if code == "auto" else code
        for is_text, actions in groupby(
            self._iter_inline_actions(source), lambda item: item[0] == "text"
        ):
            if is_text:
                text = (value for _, value in actions)
                sentences = splitter.generate_sentences(text, **{**options, "language": language})
                for chunk in self._synthesis_chunk_generator(sentences, threshold, log_chunks):
                    yield ("text", chunk)
            else:
                for kind, value in actions:
                    if kind == "language":
                        code = InlineMarkupParser.language_code(value)
                        language = options["language"] if code == "auto" else code
                    yield kind, value

    def _current_engine_voice(self, engine):
        voice = getattr(engine, "current_voice", None)
        return voice if voice is not None else getattr(engine, "voice", None)

    def _remember_inline_language_engine(self):
        if self.engine not in self._language_baselines:
            self._language_baselines[self.engine] = (
                self.engine.get_language(), self._current_engine_voice(self.engine))

    def _restore_inline_language(self):
        if not self.language_tags:
            return
        for engine, (language, voice) in self._language_baselines.items():
            try:
                if voice is not None and self._current_engine_voice(engine) is not voice:
                    engine.set_inline_voice(voice)
                if engine.get_language() != language:
                    engine.set_language(language)
            except Exception as exc:
                self._language_error = self._language_error or exc
                logging.warning("Could not restore inline language baseline: %s", exc)
        self._language_baselines.clear()
        self._synthesis_speed = self._play_default_speed
        if self.player is not None:
            self.player.speed = self._play_default_speed
        self._synthesis_language = self._play_default_language
        if self._inline_voice_baseline is not None:
            self.active_voice_tag, self.active_voice = self._inline_voice_baseline

    def _apply_inline_action(self, action_type, value):
        if action_type == "speed":
            self._synthesis_speed = value
            # Validate support before placing the event; the engine's alignment
            # and decoder taps must never see non-audio queue items.
            fmt, channels, rate = self._output_stream_info()
            try:
                probe = TempoProcessor(fmt, channels, rate, value)
                probe.cancel()
            except Exception as exc:
                self._speed_error = exc
                raise
            if not self.char_iter.immediate_stop.is_set():
                self.player.buffer_manager.audio_buffer.put(SpeedChange(value))
        elif action_type == "language":
            self._remember_inline_language_engine()
            self.engine.set_language(value)
            self._synthesis_language = value
        elif action_type == "voice":
            self._set_active_voice(*value)
        elif action_type == "pause":
            self._enqueue_silence(value[1])

    def _finish_decoded_audio(self):
        if self.char_iter.immediate_stop.is_set() or any(event.is_set() for event in self.abort_events):
            self._decoded_audio_queue.cancel()
        else:
            self._decoded_audio_queue.finish()

    def _output_stream_info(self):
        if self._alignment_binding is not None:
            return self._alignment_binding.output_format
        if self._decoded_audio_queue is not None:
            return self._decoded_audio_queue.output_format
        return self.engine.get_stream_info()

    def _aligned_input(self, source):
        for character in source:
            self._alignment_binding.append_text(character)
            yield character

    def _before_alignment_audio(self, position):
        source_position = self.tempo_timeline.output_to_source(self.player._tempo_output_offset + position)
        self.alignment.before_playback(source_position)

    def _on_alignment_progress(self, position):
        source_position = self.tempo_timeline.output_to_source(self.player._tempo_output_offset + position)
        # Do not announce a future fragment whose first sample has not played.
        source_position = max(0.0, source_position - 1.0 / self._output_stream_info()[2])
        self.alignment.advance(source_position)

    def _finish_alignment_run(self):
        if self.alignment is not None:
            self.alignment.finish()
            if self.alignment.config.mode == "export":
                self.alignment.wait()
            self.alignment.playback_finished()
            self._on_alignment_update(self.alignment.snapshot())

    def close(self):
        """Stop this stream and release its optional alignment worker."""
        self.stop()
        if self.alignment is not None:
            self.alignment.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _should_capture_alignment_audio(self, stream_format, channels, rate):
        if not self.align_words:
            return False
        if self._engine_has_native_word_timings(self.engine):
            return False
        if self.engine.can_consume_generators:
            logging.warning(
                "Forced word alignment is only available for sentence/chunk engines."
            )
            return False
        if not is_alignable_stream_format(stream_format, channels, rate):
            logging.warning(
                "Forced word alignment is unavailable for engine %s because "
                "its stream format cannot be decoded.",
                self.engine.engine_name,
            )
            return False
        return True

    @staticmethod
    def _engine_has_native_word_timings(engine):
        return bool(getattr(engine, "provides_word_timings", False))

    def _start_alignment_capture(self, sentence):
        if not self._alignment_capture_queue:
            return False
        return self._alignment_capture_queue.start_capture(sentence)

    def _finish_alignment_capture(self, capture_started):
        if not capture_started or not self._alignment_capture_queue:
            return None
        segment = self._alignment_capture_queue.finish_capture()
        if not segment or not segment.audio_bytes:
            return None
        return segment

    def _ensure_word_aligner(self):
        if self.word_aligner is None:
            from .alignment.omniasr import OmniASRCTCAligner

            self.word_aligner = OmniASRCTCAligner()
        return self.word_aligner

    def _schedule_alignment(self, segment: CapturedAudioSegment):
        aligner = self._ensure_word_aligner()
        timings_queue = self.engine.timings
        generation = self._alignment_generation

        if self.alignment_blocking:
            self._run_alignment(segment, timings_queue, aligner, generation)
            return

        self._alignment_threads = [
            thread for thread in self._alignment_threads if thread.is_alive()
        ]
        alignment_thread = threading.Thread(
            target=self._run_alignment,
            args=(segment, timings_queue, aligner, generation),
            daemon=True,
        )
        self._alignment_threads.append(alignment_thread)
        alignment_thread.start()

    def _run_alignment(
        self,
        segment: CapturedAudioSegment,
        timings_queue,
        aligner,
        generation,
    ):
        try:
            audio, sample_rate = segment.to_mono_float32()
            with self._alignment_lock:
                alignments = aligner.align_words(segment.text, audio, sample_rate)
            if generation != self._alignment_generation:
                return
            for alignment in alignments:
                if generation != self._alignment_generation:
                    return
                start_time = segment.start_time + float(alignment.start_time)
                end_time = segment.start_time + float(alignment.end_time)
                if end_time < start_time:
                    continue
                timings_queue.put(
                    TimingInfo(start_time, end_time, str(alignment.word))
                )
        except Exception as exc:
            logging.warning("Forced word alignment failed: %s", exc)

    def _extract_voice_switch_actions(self, text: str):
        """
        Backward-compatible alias for inline action extraction.
        """
        return self._extract_inline_actions(text)

    def _set_active_voice(self, tag: Optional[str], voice):
        """
        Switches the engine voice and stores the current active mapping.
        """
        logging.info(f"Switching voice to tag '{tag}'")
        if self.language_tags:
            self._remember_inline_language_engine()
            self.engine.set_inline_voice(voice)
        else:
            self.engine.set_voice(voice)
        self.active_voice_tag = tag
        self.active_voice = voice

    def _reapply_active_voice(self):
        """
        Reapplies the currently active mapped voice after an engine reload.
        """
        if self.active_voice is None:
            return

        try:
            if self.language_tags:
                self._remember_inline_language_engine()
                self.engine.set_inline_voice(self.active_voice)
            else:
                self.engine.set_voice(self.active_voice)
        except Exception as exc:
            logging.warning(
                f"Could not reapply active voice tag '{self.active_voice_tag}' "
                f"after engine switch: {exc}"
            )
            if self.language_tags:
                raise

    def _enqueue_silence(self, silence_duration: float):
        """
        Adds silence of the requested duration to the engine queue.
        """
        if silence_duration <= 0:
            return
        if self._alignment_binding is not None:
            self._alignment_binding.silence(silence_duration)
            return

        stream_format, channels, sample_rate = self._output_stream_info()
        if sample_rate <= 0 or stream_format == pyaudio.paCustomFormat:
            logging.warning(
                f"Cannot enqueue silence for engine {self.engine.engine_name} "
                "because the stream format is not PCM."
            )
            return

        silent_frames = int(sample_rate * silence_duration)
        sample_count = silent_frames * max(channels, 1)
        if stream_format == pyaudio.paInt16:
            silent_chunk = np.zeros(sample_count, dtype=np.int16)
        else:
            silent_chunk = np.zeros(sample_count, dtype=np.float32)

        silence_queue = self._decoded_audio_queue.wrapped if self._decoded_audio_queue else self.engine.queue
        silence_queue.put(silent_chunk.tobytes())

    def _is_engine_mpeg(self):
        """
        Checks if the engine is an MPEG engine.

        Returns:
            Boolean indicating if the engine is an MPEG engine.
        """
        format, channel, rate = self._output_stream_info()
        return format == pyaudio.paCustomFormat and channel == -1 and rate == -1

    def _synthesis_chunk_generator(
        self,
        generator: Iterator[str],
        buffer_threshold_seconds: float = 2.0,
        log_synthesis_chunks: bool = False,
    ) -> Iterator[str]:
        """
        Generates synthesis chunks based on buffered audio length.

        The function buffers chunks of synthesis until the buffered audio seconds fall below the provided threshold.
        Once the threshold is crossed, the buffered synthesis chunk is yielded.

        Args:
            generator: Input iterator that provides chunks for synthesis.
            buffer_threshold_seconds: Time in seconds to specify how long audio data should be buffered before yielding the synthesis chunk.
            log_synthesis_chunks: Boolean flag that, if set to True, logs the synthesis chunks to the logging system.

        Returns:
            Iterator of synthesis chunks.
        """

        # Initializes an empty string to accumulate chunks of synthesis
        synthesis_chunk = ""

        # Iterates over each chunk from the provided generator
        for chunk in generator:
            # Fetch the total seconds of buffered audio
            if self.player:
                buffered_audio_seconds = self.player.get_buffered_seconds()
            else:
                buffered_audio_seconds = 0

            # Append the current chunk (and a space) to the accumulated synthesis_chunk
            synthesis_chunk += chunk + " "

            # Check if the buffered audio is below the specified threshold
            if (
                buffered_audio_seconds < buffer_threshold_seconds
                or buffer_threshold_seconds <= 0
            ):
                # If the log_synthesis_chunks flag is True, log the current synthesis_chunk
                if log_synthesis_chunks:
                    logging.info(
                        f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:1f}s'
                    )

                # Yield the current synthesis_chunk and reset it for the next set of accumulations
                yield synthesis_chunk
                synthesis_chunk = ""

            else:
                logging.info(
                    f"summing up chunks because buffer {buffered_audio_seconds:.1f} > threshold ({buffer_threshold_seconds:.1f}s)"
                )

        # After iterating over all chunks, check if there's any remaining data in synthesis_chunk
        if synthesis_chunk:
            # If the log_synthesis_chunks flag is True, log the remaining synthesis_chunk
            if log_synthesis_chunks:
                logging.info(
                    f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:.1f}s'
                )

            # Yield the remaining synthesis_chunk
            yield synthesis_chunk
