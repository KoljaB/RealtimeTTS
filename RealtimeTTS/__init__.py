# RealtimeTTS/__init__.py

from .text_to_stream import TextToAudioStream
from .engines import BaseEngine, TimingInfo

__all__ = [
    "TextToAudioStream", "BaseEngine", "TimingInfo",
    "SystemEngine", "SystemVoice",
    "AzureEngine", "AzureVoice",
    "ElevenlabsEngine", "ElevenlabsVoice",
    "CoquiEngine", "CoquiVoice",
    "OpenAIEngine", "OpenAIVoice",
    "GTTSEngine", "GTTSVoice",
    "ParlerEngine", "ParlerVoice",
    "EdgeEngine", "EdgeVoice",
    "StyleTTSEngine", "StyleTTSVoice",
    "PiperEngine", "PiperVoice",
    "KokoroEngine", "KokoroVoice"
    "OrpheusEngine", "OrpheusVoice",
]


# Lazy loader functions for each engine group.
def _load_system_engine():
    try:
        from .engines.system_engine import SystemEngine, SystemVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load SystemEngine and SystemVoice. "
            "Please install with:\npip install realtimetts[system]"
        ) from e
    globals()["SystemEngine"] = SystemEngine
    globals()["SystemVoice"] = SystemVoice
    return SystemEngine


def _load_azure_engine():
    try:
        from .engines.azure_engine import AzureEngine, AzureVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load AzureEngine and AzureVoice. "
            "Please install with:\npip install realtimetts[azure]"
        ) from e
    globals()["AzureEngine"] = AzureEngine
    globals()["AzureVoice"] = AzureVoice
    return AzureEngine


def _load_elevenlabs_engine():
    try:
        from .engines.elevenlabs_engine import ElevenlabsEngine, ElevenlabsVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load ElevenlabsEngine and ElevenlabsVoice. "
            "Please install with:\npip install realtimetts[elevenlabs]"
        ) from e
    globals()["ElevenlabsEngine"] = ElevenlabsEngine
    globals()["ElevenlabsVoice"] = ElevenlabsVoice
    return ElevenlabsEngine


def _load_coqui_engine():
    try:
        from .engines.coqui_engine import CoquiEngine, CoquiVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load CoquiEngine and CoquiVoice. "
            "Please install with:\npip install realtimetts[coqui]"
        ) from e
    globals()["CoquiEngine"] = CoquiEngine
    globals()["CoquiVoice"] = CoquiVoice
    return CoquiEngine


def _load_openai_engine():
    try:
        from .engines.openai_engine import OpenAIEngine, OpenAIVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load OpenAIEngine and OpenAIVoice. "
            "Please install with:\npip install realtimetts[openai]"
        ) from e
    globals()["OpenAIEngine"] = OpenAIEngine
    globals()["OpenAIVoice"] = OpenAIVoice
    return OpenAIEngine


def _load_gtts_engine():
    try:
        from .engines.gtts_engine import GTTSEngine, GTTSVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load GTTSEngine and GTTSVoice. "
            "Please install with:\npip install realtimetts[gtts]"
        ) from e
    globals()["GTTSEngine"] = GTTSEngine
    globals()["GTTSVoice"] = GTTSVoice
    return GTTSEngine


def _load_parler_engine():
    try:
        from .engines.parler_engine import ParlerEngine, ParlerVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load ParlerEngine and ParlerVoice. "
            "See README for installation instructions."
        ) from e
    globals()["ParlerEngine"] = ParlerEngine
    globals()["ParlerVoice"] = ParlerVoice
    return ParlerEngine


def _load_edge_engine():
    try:
        from .engines.edge_engine import EdgeEngine, EdgeVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load EdgeEngine and EdgeVoice. "
            "Please install with:\npip install realtimetts[edge]"
        ) from e
    globals()["EdgeEngine"] = EdgeEngine
    globals()["EdgeVoice"] = EdgeVoice
    return EdgeEngine


def _load_style_engine():
    try:
        from .engines.style_engine import StyleTTSEngine, StyleTTSVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load StyleTTSEngine and StyleTTSVoice. "
            "See README for installation instructions."
        ) from e
    globals()["StyleTTSEngine"] = StyleTTSEngine
    globals()["StyleTTSVoice"] = StyleTTSVoice
    return StyleTTSEngine


def _load_piper_engine():
    try:
        from .engines.piper_engine import PiperEngine, PiperVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load PiperEngine and PiperVoice. "
            "See README for installation instructions."
        ) from e
    globals()["PiperEngine"] = PiperEngine
    globals()["PiperVoice"] = PiperVoice
    return PiperEngine


def _load_kokoro_engine():
    try:
        from .engines.kokoro_engine import KokoroEngine, KokoroVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load KokoroEngine. Make sure you have installed the kokoro dependencies by running:\n"
            "pip install realtimetts[kokoro]\n"
            "(or pip install realtimetts[kokoro,jp,zh] for japanese and chinese support)"
        ) from e
    globals()["KokoroEngine"] = KokoroEngine
    globals()["KokoroVoice"] = KokoroVoice
    return KokoroEngine


def _load_orpheus_engine():
    try:
        from .engines.orpheus_engine import OrpheusEngine, OrpheusVoice
    except ImportError as e:
        raise ImportError(
            "Failed to load OrpheusEngine and OrpheusVoice. "
            "Please install with:\npip install realtimetts[orpheus]"
        ) from e
    globals()["OrpheusEngine"] = OrpheusEngine
    globals()["OrpheusVoice"] = OrpheusVoice
    return OrpheusEngine


# Mapping names to their lazy loader functions.
_lazy_imports = {
    "SystemEngine": _load_system_engine,
    "SystemVoice": _load_system_engine,
    "AzureEngine": _load_azure_engine,
    "AzureVoice": _load_azure_engine,
    "ElevenlabsEngine": _load_elevenlabs_engine,
    "ElevenlabsVoice": _load_elevenlabs_engine,
    "CoquiEngine": _load_coqui_engine,
    "CoquiVoice": _load_coqui_engine,
    "OpenAIEngine": _load_openai_engine,
    "OpenAIVoice": _load_openai_engine,
    "GTTSEngine": _load_gtts_engine,
    "GTTSVoice": _load_gtts_engine,
    "ParlerEngine": _load_parler_engine,
    "ParlerVoice": _load_parler_engine,
    "EdgeEngine": _load_edge_engine,
    "EdgeVoice": _load_edge_engine,
    "StyleTTSEngine": _load_style_engine,
    "StyleTTSVoice": _load_style_engine,
    "PiperEngine": _load_piper_engine,
    "PiperVoice": _load_piper_engine,
    "KokoroEngine": _load_kokoro_engine,
    "KokoroVoice": _load_kokoro_engine,
    "OrpheusEngine": _load_orpheus_engine,
    "OrpheusVoice": _load_orpheus_engine,
}


def __getattr__(name):
    if name in _lazy_imports:
        return _lazy_imports[name]()
    raise AttributeError(f"module {__name__} has no attribute {name}")
