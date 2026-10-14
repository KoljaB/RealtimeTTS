"""RealtimeTTS-compatible HTTP/WebSocket server for Breeze TTS 2."""
from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path

from .engines.breeze_engine import BreezeEngine, BreezeVoice, VOCAL_TAGS
from .qwen_server import QwenHttpServer, ApiError, create_app
from .language_router import FastTextLanguageDetector


DESIGN_VOICE = "__breeze_voice_design__"
BREEZE_SAMPLING = ("cfg_scale", "seed", "max_new_tokens", "temperature",
                   "top_k", "top_p", "repetition_penalty")


class BreezeHttpServer(QwenHttpServer):
    def _validate_voice(self, value):
        if isinstance(value, str) and value.strip() == DESIGN_VOICE:
            return DESIGN_VOICE
        return super()._validate_voice(value)

    def __init__(self, engine, *, diagnostics_worker_state=None, diagnostics_model=None, **kwargs):
        self._diagnostics_worker_state = diagnostics_worker_state
        self._diagnostics_model = diagnostics_model
        kwargs.setdefault("alias", "breeze-tts-2-q8")
        kwargs.setdefault("comma_silence_duration", 0.0)
        kwargs.setdefault("sentence_silence_duration", 0.0)
        super().__init__(engine, **kwargs)
        self._sampling_defaults["cfg_scale"] = engine.cfg_scale
        self._failure_dir = self.registry.root.parent / "failures"

    def record_synthesis_failure(self, request, state):
        from .breeze_diagnostics import write_failure
        return write_failure(self._failure_dir, request, state,
                             worker_state=self._diagnostics_worker_state,
                             model_path=self._diagnostics_model)

    def synthesis_voice(self, name, language, instructions=None):
        if name == DESIGN_VOICE:
            return BreezeVoice("Voice Design", instruct=instructions,
                               language=language, voice_design=True)
        record = self.registry.get(name)
        if record is None:
            raise RuntimeError(f"registered voice disappeared: {name}")
        if record.kind != "wav":
            raise ValueError("Breeze accepts reference WAVs, not Qwen latent files")
        return BreezeVoice(name, self.registry._resolve_file(record.files[0]),
                           record.ref_text, instructions, language)

    def _parse_voice_upload(self, payload):
        result = super()._parse_voice_upload(payload)
        if result[0] == DESIGN_VOICE:
            raise ApiError(400, "invalid_request_error", "This name is reserved for voice design")
        if result[2] != "wav" or not result[1]:
            raise ApiError(400, "invalid_request_error",
                           "Breeze voice registration needs wav_b64 and exact ref_text")
        return result

    def parse_speech_options(self, payload):
        for name in ("clone_mode", "do_sample", "subtalker_do_sample",
                     "subtalker_temperature", "subtalker_top_k", "subtalker_top_p"):
            if name in payload:
                raise ApiError(400, "unsupported_operation", f"Breeze does not support {name}")
        # Native Breeze uses zero to select its default sampling values.
        zero_defaults = {}
        parent_payload = dict(payload)
        for name in ("top_p", "repetition_penalty"):
            value = payload.get(name)
            if not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0:
                zero_defaults[name] = 0.0
                parent_payload.pop(name)
        options = super().parse_speech_options(parent_payload)
        options.sampling.update(zero_defaults)
        if options.voice == DESIGN_VOICE and not options.instructions:
            raise ApiError(400, "invalid_request_error", "Voice design requires a voice description")
        if "cfg_scale" in payload:
            value = payload["cfg_scale"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 10:
                raise ApiError(400, "invalid_request_error", "cfg_scale must be between 1 and 10")
            options.sampling["cfg_scale"] = float(value)
        return options

    def capabilities(self):
        result = super().capabilities()
        result["engine"].update(backend="Breeze-TTS-2.cpp", clone_mode="reference", clone_modes=[])
        result["features"].update(direct_voice_control=True, vocal_tags=list(VOCAL_TAGS),
                                  voice_design=True, voice_design_voice=DESIGN_VOICE,
                                  sampling_parameters=list(BREEZE_SAMPLING),
                                  per_text_instructions=True, supported_languages=["en", "zh"],
                                  alignment_non_speech_ranges="utf16_code_units",
                                  reference_prefix_cache=bool(
                                      getattr(self.engine, "reference_prefix_cache", False)
                                  ))
        result["stream_controls"]["cancellation"] = "native_frame_checkpoint"
        result["diagnostics"] = {"last_synthesis": dict(getattr(self.engine, "last_metrics", {}))}
        return result


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    worker = parser.add_mutually_exclusive_group()
    worker.add_argument("--model", type=Path,
                        help="Start the optional packaged worker with a local Q8_0 GGUF")
    worker.add_argument("--worker-url", help="Connect to an existing native worker")
    parser.add_argument("--worker-ws-url")
    parser.add_argument("--voice", default="sl_ref1")
    parser.add_argument("--voice-reference", type=Path)
    parser.add_argument("--voice-reference-text")
    parser.add_argument("--voice-dir", type=Path, default=Path(".breeze-voices"))
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--reference-prefix-cache", action="store_true")
    parser.add_argument("--language-id-model", type=Path,
                        help="Local fastText lid.176.bin for background text classification")
    parser.add_argument("--alignment-model", type=Path)
    parser.add_argument("--alignment-language")
    parser.add_argument("--alignment-threads", type=int, default=2)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--diagnostics-worker-state", type=Path)
    parser.add_argument("--diagnostics-model", type=Path)
    return parser


def main(argv=None):
    import os
    from contextlib import nullcontext
    import uvicorn
    from .qwen_server import _is_loopback_host
    args = build_argument_parser().parse_args(argv)
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if not _is_loopback_host(args.host) and not key:
        raise SystemExit("Non-loopback Breeze binding requires --api-key-env with a nonempty key")
    if bool(args.voice_reference) != bool(args.voice_reference_text):
        raise SystemExit("Pass both --voice-reference and --voice-reference-text")
    if args.model and args.worker_ws_url:
        raise SystemExit("--model cannot be combined with --worker-ws-url")
    if args.model:
        from .breeze_worker import local_worker
        worker = local_worker(args.model, args.voice_dir / "native")
    else:
        worker = nullcontext((args.worker_url or "http://127.0.0.1:8137",
                              args.worker_ws_url or "ws://127.0.0.1:8138"))
    with worker as (worker_url, worker_ws_url):
        print("Connecting to the local Breeze GPU worker...", flush=True)
        engine = BreezeEngine(base_url=worker_url, ws_url=worker_ws_url,
                              cfg_scale=args.cfg_scale, seed=args.seed,
                              max_new_tokens=args.max_new_tokens,
                              reference_prefix_cache=args.reference_prefix_cache,
                              alignment_model=args.alignment_model,
                              alignment_language=args.alignment_language,
                              alignment_threads=args.alignment_threads)
        server = BreezeHttpServer(
            engine, voice_dir=args.voice_dir, api_key=key,
            language_detector=(FastTextLanguageDetector(args.language_id_model)
                               if args.language_id_model is not None else None),
            diagnostics_worker_state=args.diagnostics_worker_state,
            diagnostics_model=args.diagnostics_model,
            startup_warmup_voice=None if args.no_warmup else args.voice,
            startup_warmup_routes=False,
        )
        if args.voice_reference:
            print(f"Registering reference voice {args.voice}...", flush=True)
            server.register_voice({
                "name": args.voice, "ref_text": args.voice_reference_text,
                "wav_b64": base64.b64encode(args.voice_reference.read_bytes()).decode("ascii"),
            })
        if server.registry.get(args.voice) is None:
            server.startup_warmup_voice = None
            print("Register a reference voice or use Voice Design in Studio.", flush=True)
        print(f"Starting Breeze at http://{args.host}:{args.port}; warming before readiness...", flush=True)
        uvicorn.run(create_app(server), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
