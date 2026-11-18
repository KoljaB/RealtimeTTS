import base64
import io
import json
import unittest
import importlib.util
import sys
import types
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import wave
from unittest.mock import patch
from urllib.error import HTTPError, URLError

# Keep transport/PCM checks runnable without the optional playback wheel.
# Load the real BaseEngine (including its metaclass and cancellation event).
if importlib.util.find_spec("pyaudio") is None:
    backend = types.ModuleType("RealtimeTTS._audio_backend")
    backend.pyaudio = types.SimpleNamespace(paInt16=8)
    sys.modules["RealtimeTTS._audio_backend"] = backend
    try:
        from RealtimeTTS.engines import sixtydb_engine as module
    finally:
        sys.modules.pop("RealtimeTTS._audio_backend", None)
else:
    from RealtimeTTS.engines import sixtydb_engine as module
SixtyDBEngine, SixtyDBVoice = module.SixtyDBEngine, module.SixtyDBVoice


PCM = b"\x01\x00\xff\xff\x03\x00"


def wav_bytes(pcm=PCM, rate=24000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return output.getvalue()


def encoded(pcm):
    return base64.b64encode(pcm).decode("ascii")


class Response(io.BytesIO):
    def __init__(self, body, content_type="application/x-ndjson", headers=None):
        super().__init__(body)
        self.headers = {"Content-Type": content_type, **(headers or {})}


class SixtyDBTests(unittest.TestCase):
    def engine(self, **kwargs):
        return SixtyDBEngine(voice="workspace-voice", api_key="test-key", **kwargs)

    def run_response(self, response, engine=None):
        engine = engine or self.engine()
        with patch.object(module, "urlopen", return_value=response) as opened:
            result = engine.synthesize("Hello")
        return engine, result, opened

    def test_ndjson_contract_double_encoding_and_raw_pcm(self):
        first = encoded(json.dumps({"result": {"audioContent": encoded(PCM[:2])}}).encode())
        body = b"\n".join(json.dumps(record).encode() for record in [
            {"type": "meta", "sample_rate": 24000, "encoding": "LINEAR16"},
            {"result": {"audioContent": first}},
            {"result": {"audioContent": encoded(PCM[2:])}},
        ])
        engine, result, opened = self.run_response(Response(body), self.engine(model="60db-fast-v01", speed=1.5))
        self.assertTrue(result)
        self.assertEqual(engine.queue.get_nowait() + engine.queue.get_nowait(), PCM)
        self.assertAlmostEqual(engine.audio_duration, len(PCM) / 48000)
        self.assertEqual(engine.get_stream_info(), (module.pyaudio.paInt16, 1, 24000))
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.60db.ai/tts-synthesize")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {
            "text": "Hello", "voice_id": "workspace-voice", "speed": 1.5,
            "model_id": "60db-fast-v01", "output_format": "wav", "timestamp_type": "NONE",
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
        })
        self.assertTrue(opened.call_args.kwargs["timeout"] > 0)

    def test_binary_wav_and_json_wav_strip_headers(self):
        for response in [Response(wav_bytes(), "audio/wav"), Response(json.dumps({
            "success": True, "audio_base64": encoded(wav_bytes()), "output_format": "wav",
        }).encode(), "application/json")]:
            with self.subTest(content_type=response.headers):
                engine, result, _ = self.run_response(response)
                self.assertTrue(result)
                self.assertEqual(engine.queue.get_nowait(), PCM)

    def test_raw_pcm_starting_with_brace_is_preserved(self):
        pcm = b"{\x00\xff\xff"
        engine, result, _ = self.run_response(Response(json.dumps({"audioContent": encoded(pcm)}).encode()))
        self.assertTrue(result)
        self.assertEqual(engine.queue.get_nowait(), pcm)

    def test_invalid_audio_and_error_responses_fail(self):
        responses = [
            Response(b""), Response(b'{"type":"meta"}'),
            Response(b'{"success":false}'), Response(b'{"type":"error"}'),
            Response(b'{"result":{"success":false}}'),
            Response(b'{"backendResponse":{"success":false}}', "application/json"),
            Response(b'{"audioContent":"bad!"}'), Response(b'{"audioContent":"AA=="}'),
            Response(b'{"audioContent":"", "encoding":"mp3"}'),
            Response(b'{"sample_rate":48000}'), Response(b'{"audio_config":{"audio_encoding":"OGG_OPUS"}}'),
            Response(b'{invalid json'), Response(b"[]", "application/json"),
            Response(wav_bytes(rate=48000), "audio/wav"),
            Response(wav_bytes(channels=2), "audio/wav"),
            Response(wav_bytes()[:-2], "audio/wav"),
            Response(b"ID3notpcm", "application/octet-stream"),
            Response(PCM, "audio/mpeg"), Response(PCM, "audio/l16"),
            Response(PCM, "audio/pcm", {"X-Sample-Rate": "48000"}),
        ]
        for response in responses:
            with self.subTest(headers=response.headers, body=response.getvalue()):
                engine, result, _ = self.run_response(response)
                self.assertFalse(result)
                self.assertIsNotNone(engine.last_error)
                self.assertTrue(engine.queue.empty())

    def test_partial_error_is_not_success(self):
        body = json.dumps({"audioContent": encoded(PCM)}).encode() + b'\n{"type":"error"}'
        engine, result, _ = self.run_response(Response(body))
        self.assertFalse(result)
        self.assertEqual(engine.queue.get_nowait(), PCM)
        self.assertIsNotNone(engine.last_error)

    def test_network_errors_fail(self):
        for error in [URLError("offline"), HTTPError(module.__name__, 401, "Unauthorized", {}, None), TimeoutError()]:
            engine = self.engine()
            with patch.object(module, "urlopen", side_effect=error):
                self.assertFalse(engine.synthesize("Hello"))
            self.assertIs(engine.last_error, error)
            self.assertTrue(engine.queue.empty())

    def test_cancellation_returns_false_and_response_context_closes(self):
        engine = self.engine()
        response = Response(PCM, "audio/pcm")
        original_read = response.read
        def stop_after_read(size):
            audio = original_read(size)
            engine.stop()
            return audio
        response.read = stop_after_read
        engine, result, _ = self.run_response(response, engine)
        self.assertFalse(result)
        self.assertTrue(response.closed)
        self.assertTrue(engine.queue.empty())

    def test_stalled_http_ndjson_cancellation_returns_false_and_closes_safely(self):
        partial_sent = threading.Event()
        release_server = threading.Event()
        read_started = threading.Event()
        stop_finished = threading.Event()
        responses, results, errors = [], [], []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                self.wfile.write(b'{"result":')
                self.wfile.flush()
                partial_sent.set()
                release_server.wait(3)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        serving.start()
        engine = self.engine(timeout=0.5)
        engine.base_url = "http://127.0.0.1:" + str(server.server_port)
        original_urlopen = module.urlopen

        def capture_response(*args, **kwargs):
            response = original_urlopen(*args, **kwargs)
            responses.append(response)
            original_readline = response.readline
            def readline(*args, **kwargs):
                read_started.set()
                return original_readline(*args, **kwargs)
            response.readline = readline
            return response

        def synthesize():
            try:
                results.append(engine.synthesize("Hello"))
            except Exception as exc:
                errors.append(exc)

        def cancel():
            try:
                engine.stop()
            except Exception as exc:
                errors.append(exc)
            finally:
                stop_finished.set()

        synthesis = threading.Thread(target=synthesize)
        cancellation = threading.Thread(target=cancel)
        try:
            with patch.object(module, "urlopen", side_effect=capture_response):
                synthesis.start()
                self.assertTrue(partial_sent.wait(2))
                self.assertTrue(read_started.wait(2))
                cancellation.start()
                # stop() only signals the event; response ownership stays with
                # the synthesis thread while its real socket read times out.
                self.assertTrue(stop_finished.wait(0.1))
                synthesis.join(2)
                self.assertFalse(synthesis.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, [False])
            self.assertTrue(engine.queue.empty())
            self.assertTrue(responses[0].closed)
            self.assertIsInstance(engine.last_error, TimeoutError)
        finally:
            release_server.set()
            engine.stop()
            if synthesis.ident is not None:
                synthesis.join(2)
            if cancellation.ident is not None:
                cancellation.join(2)
            server.shutdown()
            server.server_close()
            serving.join(2)

    def test_voice_listing_and_selection(self):
        engine = self.engine()
        response = Response(json.dumps({"success": True, "data": [{
            "voice_id": "real-workspace-id", "name": "Narrator", "model": "60db Fast", "labels": {"language": "en"},
        }]}).encode(), "application/json")
        with patch.object(module, "urlopen", return_value=response) as opened:
            voices = engine.get_voices(model="fast")
        self.assertEqual(opened.call_args.args[0].full_url, "https://api.60db.ai/voices?model=fast")
        self.assertEqual(voices, [SixtyDBVoice("real-workspace-id", "Narrator", "en", "60db Fast")])
        engine.set_voice(voices[0])
        self.assertEqual(engine.voice, "real-workspace-id")
        self.assertEqual(repr(voices[0]), "Narrator")

    def test_validation_and_no_import_or_constructor_requests(self):
        with patch.object(module, "urlopen") as opened:
            engine = self.engine()
            self.assertFalse(engine.synthesize(" "))
            self.assertFalse(engine.synthesize("x" * 5001))
            for voice in [None, "", 123]:
                with self.assertRaises(ValueError):
                    engine.set_voice(voice)
            for speed in [0.49, 2.01, float("nan"), float("inf"), True, "1"]:
                with self.assertRaises(ValueError):
                    engine.set_voice_parameters(speed=speed)
            with self.assertRaises(ValueError):
                engine.set_voice_parameters(pitch=1)
            with self.assertRaises(ValueError):
                engine.get_voices(model="invalid")
            opened.assert_not_called()
        with patch.dict("os.environ", {"SIXTYDB_API_KEY": "env-test-key"}, clear=True):
            self.assertEqual(SixtyDBEngine(voice="id").api_key, "env-test-key")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                SixtyDBEngine(voice="id")
        for timeout in [0, float("nan"), True]:
            with self.assertRaises(ValueError):
                self.engine(timeout=timeout)

    def test_public_lazy_exports(self):
        import RealtimeTTS
        import RealtimeTTS.engines
        self.assertIs(RealtimeTTS.SixtyDBEngine, SixtyDBEngine)
        self.assertIs(RealtimeTTS.SixtyDBVoice, SixtyDBVoice)
        self.assertIs(RealtimeTTS.engines.SixtyDBEngine, SixtyDBEngine)
        self.assertIs(RealtimeTTS.engines.SixtyDBVoice, SixtyDBVoice)


if __name__ == "__main__":
    unittest.main()
