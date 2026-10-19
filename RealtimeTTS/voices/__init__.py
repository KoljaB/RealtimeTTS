"""Reference recordings shipped with RealtimeTTS (no network or model loading)."""

from __future__ import annotations

import atexit
import json
from contextlib import ExitStack
from functools import lru_cache
from importlib import resources
from pathlib import Path

_resource_files = ExitStack()
atexit.register(_resource_files.close)


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict, ...]:
    data = json.loads(resources.files(__name__).joinpath("catalog.json").read_text(encoding="utf-8"))
    return tuple(data["voices"])


@lru_cache(maxsize=None)
def _audio_path(filename: str) -> Path:
    # Wheels installed by pip have real files. as_file also keeps an extracted
    # resource alive for the process lifetime when the package is zip-imported.
    resource = resources.files(__name__)
    for part in filename.split("/"):
        resource = resource.joinpath(part)
    return _resource_files.enter_context(resources.as_file(resource))


def _voice_data(entry: dict) -> dict:
    return {
        "name": entry["name"],
        "ref_audio": _audio_path(entry["file"]),
        "ref_text": entry["ref_text"],
        "language": entry["language"],
    }


def bundled_voices_data(engine: str) -> list[dict]:
    """Return fresh voice constructor arguments for the requested engine."""
    return [_voice_data(entry) for entry in _catalog() if entry["engine"] == engine]


def bundled_voice_data(engine: str, language: str) -> dict:
    """Resolve a language code, language name or bundled voice name."""
    name = str(language).strip().lower()
    entries = [entry for entry in _catalog() if entry["engine"] == engine]
    for entry in entries:
        if name in (entry["name"], entry["language_code"], entry["language"]):
            return _voice_data(entry)
    available = ", ".join(entry["language_code"] for entry in entries)
    raise ValueError(f"No bundled {engine} voice for {language!r}; available languages: {available}")


def register_bundled_voices(registry, engine: str) -> None:
    """Populate the server registry without loading every voice into the model.

    Existing names are user-owned and remain unchanged. References are encoded
    lazily by the engine on selection; startup may warm the selected default.
    """
    for voice in bundled_voices_data(engine):
        if registry.get(voice["name"]) is not None:
            continue
        record = registry.candidate(voice["name"], voice["ref_text"], "wav",
                                    [voice["ref_audio"].read_bytes()])
        try:
            registry.commit(record)
        except BaseException:
            registry.discard_candidate(record)
            raise
