from RealtimeTTS import TextToAudioStream, CoquiEngine
from rvc.realtimervc import RealtimeRVC
from bufferstream import BufferStream
from pathlib import Path
import logging
import time
import os


class XTTSRVCSynthesizer:
    def __init__(
        self,
        xtts_model: str = None,
        xtts_voice: str = None,
        rvc_model: str = None,
        rvc_sample_rate: int = 40000,
        use_logging=False,
    ):
        """
        Initializes the realtime RVC synthesizer.

        Args:
            xtts_model (str): Path to the folder containing the xtts files
                (model.pth, config.json, vocab.json, speakers_xtts.pth)
            xtts_voice (str): Path to the file with the ~5-10 second mono 22050 Hz
                referecence voice wave file
            rvc_model (str): Path to the .pth file with the rvc reference.
                The .index file should be in the same folder and have the same
                name like this .pth file it belongs to.
            rvc_sample_rate (int): Mostly 40000 or 48000. The sample rate
                the rvc model was trained against.
            use_logging (bool): Usage of extended debug logging.
        """

        import logging

        level = logging.DEBUG if use_logging else logging.WARNING
        logging.basicConfig(level=level)

        self.xtts_model_loaded = False
        self.buffer = BufferStream()
        self.use_logging = use_logging
        self.xtts_voice = xtts_voice
        self.engine = None

        if self.use_logging:
            print("Extended logging")

        self.rvc = None
        if rvc_model is not None:
            self.load_rvc_model(rvc_model, rvc_sample_rate)

        if self.use_logging:
            print("Loading XTTS model")
        self.load_xtts_model(xtts_model)

    def push_text(self, text: str):
        self.buffer.add(text)
        if not self.xtts_model_loaded:
            self.load_xtts_model()

        self.ensure_playing()

    def ensure_playing(self):
        def on_audio_chunk(chunk):
            _, _, sample_rate = self.engine.get_stream_info()
            self.rvc.feed(chunk, sample_rate)

        if not self.stream.is_playing():
            self.stream.feed(self.buffer.gen())
            play_params = {
                "fast_sentence_fragment": True,
                "log_synthesized_text": True,
                "minimum_sentence_length": 10,
                "minimum_first_fragment_length": 10,
                "context_size": 4,
                "sentence_fragment_delimiters": '.?!;:,\n()[]{}。-“”„”—…/|《》¡¿"',
                "force_first_fragment_after_words": 9999999,
            }

            if self.rvc is not None:
                play_params["on_audio_chunk"] = on_audio_chunk
                play_params["muted"] = True

            self.stream.play_async(**play_params)

    def synthesize(self):
        self.ensure_playing()
        self.buffer.stop()

        while self.stream.is_playing():
            time.sleep(0.01)

        self.buffer = BufferStream()

    def load_xtts_model(self, local_path: str = None):
        if self.use_logging:
            if not local_path:
                print("loading default xtts model")
            else:
                print(f"loading xtts model {local_path}")

        level = logging.DEBUG if self.use_logging else logging.WARNING
        voice = self.xtts_voice if self.xtts_voice else ""

        if not self.engine:
            engine_params = {
                "language": "en",
                "level": level,
                "voice": voice,
                "speed": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 10,
                "top_k": 70,
                "top_p": 0.9,
                "add_sentence_filter": True,
                "comma_silence_duration": 0.1,
                "sentence_silence_duration": 0.3,
                "default_silence_duration": 0.1,
            }

            if local_path is not None:
                # Verify that local_path is a folder name
                if not os.path.isdir(local_path):
                    raise ValueError(
                        f"local_path must be a valid directory: {local_path}"
                    )
                xtts_model_name = os.path.basename(local_path)
                xtts_model_path = os.path.dirname(local_path)

                engine_params["specific_model"] = xtts_model_name
                engine_params["local_models_path"] = xtts_model_path

            self.engine = CoquiEngine(**engine_params)
            self.stream = TextToAudioStream(self.engine)

            print("Performing warmup")
            if self.rvc:
                self.rvc.set_muted()
            self.stream.feed("warmup")
            if self.rvc is not None:

                def on_audio_chunk(chunk):
                    _, _, sample_rate = self.engine.get_stream_info()
                    self.rvc.feed(chunk, sample_rate)

                self.stream.play(muted=True, on_audio_chunk=on_audio_chunk)
            else:
                self.stream.play(muted=True)
            if self.rvc:
                self.rvc.set_muted(False)
            print("Warmup performed")

        else:
            if voice and len(voice) > 0:
                self.engine.set_voice(voice)

            if local_path is not None:
                if not os.path.isdir(local_path):
                    raise ValueError(
                        f"local_path must be a valid directory: {local_path}"
                    )
                xtts_model_name = os.path.basename(local_path)
                xtts_model_path = os.path.dirname(local_path)

                self.engine.local_models_path = xtts_model_path
                self.engine.set_model(xtts_model_name)

        self.xtts_model_loaded = True

    def wait_playing(self):
        while self.stream.is_playing():
            time.sleep(0.01)

        if self.rvc:
            while self.rvc.is_playing():
                time.sleep(0.01)

    def stop(self):
        self.buffer.stop()

        if self.stream.is_playing():
            print("Stopping stream")
            self.stream.stop()

        if self.rvc.is_playing():
            print("Stopping rvc")
            self.rvc.stop()

        self.wait_playing()
        self.buffer = BufferStream()

    def load_rvc_model(self, rvc_model: str = None, rvc_sample_rate=40000):
        rvc_model_name = os.path.basename(rvc_model)
        rvc_model_name = str(Path(rvc_model_name).with_suffix(""))
        rvc_model_path = os.path.dirname(rvc_model)

        if self.rvc is None:
            self.rvc = RealtimeRVC(rvc_model_path, sample_rate=rvc_sample_rate)
            self.rvc.start(rvc_model_name)
        else:
            self.rvc.rvc_model_path = rvc_model_path
            self.rvc.set_model(rvc_model_name)

    def shutdown(self):
        if self.rvc:
            self.rvc.shutdown()
        if self.stream is not None and self.stream.is_playing():
            self.stream.stop()
        if self.engine is not None:
            self.engine.shutdown()
