"""Language-routing HTTP/WebSocket facade for configured TTS providers."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from .engines.language_routing_engine import LanguageRoute, LanguageRoutingEngine, normalize_language
from .engines.remote_tts_engine import RemoteTTSEngine
from .language_router import FastTextLanguageDetector, normalize_qwen_language
from .qwen_server import ApiError, QwenHttpServer, create_app


class LanguageRoutingServer(QwenHttpServer):
    """Reuse the streaming protocol, with voices and controls bound by route."""

    def __init__(self, engine, *, providers=None, **kwargs):
        self.providers = dict(providers or {})
        self.default_language = engine.routing_metadata()["default_language"]
        self._voices = {}
        for entries in engine.routes.values():
            for route in entries:
                provider = self._provider_id(route.engine)
                if route.voice is not None:
                    self._voices.setdefault("default", {})[provider] = self._voice_name(route.voice)
                for name, voice in (route.voices or {}).items():
                    self._voices.setdefault(name, {})[provider] = self._voice_name(voice)
        if not self._voices:
            self._voices["default"] = {}
        kwargs.setdefault("alias", "language-routing")
        kwargs.setdefault("language", self.default_language)
        super().__init__(engine, **kwargs)
        self._sampling_defaults.clear()

    def _provider_id(self, engine):
        for name, provider in self.providers.items():
            if provider is engine:
                return name
        return str(getattr(engine, "engine_name", type(engine).__name__))

    @staticmethod
    def _voice_name(voice):
        return voice if isinstance(voice, str) else str(getattr(voice, "name", type(voice).__name__))

    def list_voices(self):
        return [{"name": name, "kind": "routed", "providers": mapping}
                for name, mapping in self._voices.items()]

    def _validate_voice(self, value):
        if not isinstance(value, str) or value not in self._voices:
            raise ApiError(400, "invalid_request_error", "voice must name a configured logical voice")
        return value

    def register_voice(self, payload):
        raise ApiError(400, "unsupported_operation", "Register voices on each provider and configure their logical mappings")

    def remove_voice(self, name):
        raise ApiError(400, "unsupported_operation", "Remove voice mappings in the routing configuration")

    def parse_speech_options(self, payload):
        options = super().parse_speech_options(payload)
        if options.sampling:
            raise ApiError(400, "unsupported_operation", "Configure sampling separately on each provider")
        return options

    def configure_engine(self, request):
        self.engine.set_language(request.language)
        self.engine.set_voice(request.voice)
        self.engine.set_instructions(request.instructions or "")

    def route_request(self, request, *, fallback_language=None):
        detection = None
        if request.language == "auto":
            language = fallback_language or self.default_language
            if self.language_detector is not None:
                detection = self.language_detector.detect(request.text)
                language = detection.language
        else:
            language = request.language
        return replace(request, language=normalize_qwen_language(language)), detection

    def warmup_voice(self, voice_name, *, language=None):
        # Reference preparation belongs to each already-running provider.
        self._validate_voice(voice_name)

    def capabilities(self):
        result = super().capabilities()
        result["engine"] = {"name": "language-routing", "backend": "configured-providers"}
        metadata = self.engine.routing_metadata()
        metadata["routes"] = {
            language: [self._provider_id(route.engine) for route in entries]
            for language, entries in self.engine.routes.items()
        }
        metadata["providers"] = {
            self._provider_id(route.engine): {
                "direct_voice_control": bool(getattr(route.engine, "direct_voice_control", False)),
                "word_timings": bool(getattr(route.engine, "provides_word_timings", False)),
            }
            for entries in self.engine.routes.values() for route in entries
        }
        designed = []
        for name, mappings in self._voices.items():
            for provider_name, mapped in mappings.items():
                provider = self.providers.get(provider_name)
                if mapped == getattr(provider, "features", {}).get("voice_design_voice"):
                    designed.append(name)
        metadata["voice_design_voices"] = sorted(set(designed))
        result["routing"] = metadata
        result["features"].update(
            language_engine_routing=True, per_text_language=True, per_text_voice=True,
            direct_voice_control=self.engine.direct_voice_control,
            per_text_instructions=self.engine.direct_voice_control,
            voice_cloning=False, voice_design=bool(designed), model_type="routing",
            voice_design_voice=designed[0] if designed else None,
            sampling_parameters=[],
            mixed_language_fallback=isinstance(self.language_detector, FastTextLanguageDetector),
            streaming_alignment=self.engine.provides_word_timings,
            alignment_timestamps=["word"] if self.engine.provides_word_timings else [],
        )
        return result


def _credential(provider):
    variable, filename = provider.get("api_key_env"), provider.get("api_key_file")
    if variable and filename:
        raise ValueError("Use either api_key_env or api_key_file for a provider")
    if variable:
        value = os.environ.get(variable)
        if not value:
            raise ValueError("A configured provider credential environment variable is empty")
        return value
    return Path(filename).expanduser().read_text(encoding="utf-8").strip() if filename else None


def server_from_config(config, *, voice_dir, api_key=None):
    """Build explicit remote routes; no provider model or sampling defaults change."""
    allowed = {"default_language", "providers", "routes", "silence",
               "language_detector_model", "pre_first_pcm_timeout"}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ValueError("Unknown routing configuration fields")
    configured = config.get("providers")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("providers must be a non-empty object")
    providers, bindings = {}, {}
    for name, values in configured.items():
        if not isinstance(values, dict) or values.keys() - {
            "url", "voice", "voices", "api_key_env", "api_key_file", "instructions", "timeout"
        }:
            raise ValueError("Unknown provider configuration fields")
        provider = RemoteTTSEngine(
            values["url"], values["voice"], api_key=_credential(values),
            instructions=values.get("instructions"), timeout=values.get("timeout", 120),
        )
        voice_map = values.get("voices", {})
        if (not isinstance(voice_map, dict)
                or any(not isinstance(k, str) or not k.strip()
                       or not isinstance(v, str) or not v.strip() for k, v in voice_map.items())):
            raise ValueError("Provider voices must map logical names to provider voice names")
        known_voices = set(provider.get_voices())
        design_voice = provider.features.get("voice_design_voice")
        if design_voice:
            known_voices.add(design_voice)
        if any(voice not in known_voices for voice in [values["voice"], *voice_map.values()]):
            raise ValueError("A configured provider voice is not registered")
        providers[name] = provider
        bindings[name] = LanguageRoute(provider, voice=values["voice"], voices=voice_map)
    routes = config.get("routes")
    if not isinstance(routes, dict) or not routes:
        raise ValueError("routes must be a non-empty object")
    normalized = {}
    for language, names in routes.items():
        if not isinstance(names, list) or not names or any(name not in bindings for name in names):
            raise ValueError("Each route must list configured provider names")
        canonical = normalize_language(language)
        if canonical in normalized:
            raise ValueError("Duplicate canonical language route")
        normalized[canonical] = [bindings[name] for name in names]
    router = LanguageRoutingEngine(
        normalized, default_language=config.get("default_language", "en"),
        pre_first_pcm_timeout=config.get("pre_first_pcm_timeout", 30.0),
    )
    detector = None
    if config.get("language_detector_model"):
        detector = FastTextLanguageDetector(
            model_path=config["language_detector_model"],
            fallback_language=config.get("default_language", "en"),
        )
    silence = config.get("silence", {})
    if not isinstance(silence, dict) or silence.keys() - {"comma", "sentence", "transition"}:
        raise ValueError("silence must contain only comma, sentence, and transition")
    return LanguageRoutingServer(
        router, providers=providers, voice_dir=Path(voice_dir), api_key=api_key,
        language_detector=detector,
        comma_silence_duration=silence.get("comma", .15),
        sentence_silence_duration=silence.get("sentence", .30),
        language_transition_silence_duration=silence.get("transition", .08),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".cache/realtimetts/routing")
    parser.add_argument("--api-key-env", default="REALTIMETTS_API_KEY")
    args = parser.parse_args()
    from .qwen_server import _is_loopback_host
    api_key = os.environ.get(args.api_key_env)
    if not _is_loopback_host(args.host) and not api_key:
        parser.error("Non-loopback routing requires --api-key-env with a nonempty key")
    server = server_from_config(
        json.loads(args.config.read_text(encoding="utf-8")), voice_dir=args.state_dir,
        api_key=api_key,
    )
    import uvicorn
    uvicorn.run(create_app(server), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
