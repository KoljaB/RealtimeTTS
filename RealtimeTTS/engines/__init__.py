from .base_engine import BaseEngine, TimingInfo

__all__ = [
    "BaseEngine", "TimingInfo",
    "AzureEngine", "AzureVoice",
    "SystemEngine", "SystemVoice",
    "ElevenlabsEngine", "ElevenlabsVoice",
    "CoquiEngine", "CoquiVoice",
    "OpenAIEngine", "OpenAIVoice",
    "GTTSEngine", "GTTSVoice",
    "ParlerEngine", "ParlerVoice",
    "EdgeEngine", "EdgeVoice",
    "StyleTTSEngine", "StyleTTSVoice",
    "PiperEngine", "PiperVoice",
    "KokoroEngine", "KokoroVoice",
    "OrpheusEngine", "OrpheusVoice",
    "ZipVoiceEngine", "ZipVoiceVoice",
]


# Lazy loader functions for the engines in this subpackage.
def _load_azure_engine():
    from .azure_engine import AzureEngine, AzureVoice
    globals()["AzureEngine"] = AzureEngine
    globals()["AzureVoice"] = AzureVoice
    return AzureEngine


def _load_system_engine():
    from .system_engine import SystemEngine, SystemVoice
    globals()["SystemEngine"] = SystemEngine
    globals()["SystemVoice"] = SystemVoice
    return SystemEngine


def _load_elevenlabs_engine():
    from .elevenlabs_engine import ElevenlabsEngine, ElevenlabsVoice
    globals()["ElevenlabsEngine"] = ElevenlabsEngine
    globals()["ElevenlabsVoice"] = ElevenlabsVoice
    return ElevenlabsEngine


def _load_coqui_engine():
    from .coqui_engine import CoquiEngine, CoquiVoice
    globals()["CoquiEngine"] = CoquiEngine
    globals()["CoquiVoice"] = CoquiVoice
    return CoquiEngine


def _load_openai_engine():
    from .openai_engine import OpenAIEngine, OpenAIVoice
    globals()["OpenAIEngine"] = OpenAIEngine
    globals()["OpenAIVoice"] = OpenAIVoice
    return OpenAIEngine


def _load_gtts_engine():
    from .gtts_engine import GTTSEngine, GTTSVoice
    globals()["GTTSEngine"] = GTTSEngine
    globals()["GTTSVoice"] = GTTSVoice
    return GTTSEngine


def _load_parler_engine():
    from .parler_engine import ParlerEngine, ParlerVoice
    globals()["ParlerEngine"] = ParlerEngine
    globals()["ParlerVoice"] = ParlerVoice
    return ParlerEngine


def _load_edge_engine():
    from .edge_engine import EdgeEngine, EdgeVoice
    globals()["EdgeEngine"] = EdgeEngine
    globals()["EdgeVoice"] = EdgeVoice
    return EdgeEngine


def _load_style_engine():
    from .style_engine import StyleTTSEngine, StyleTTSVoice
    globals()["StyleTTSEngine"] = StyleTTSEngine
    globals()["StyleTTSVoice"] = StyleTTSVoice
    return StyleTTSEngine


def _load_piper_engine():
    from .piper_engine import PiperEngine, PiperVoice
    globals()["PiperEngine"] = PiperEngine
    globals()["PiperVoice"] = PiperVoice
    return PiperEngine


def _load_kokoro_engine():
    from .kokoro_engine import KokoroEngine, KokoroVoice
    globals()["KokoroEngine"] = KokoroEngine
    globals()["KokoroVoice"] = KokoroVoice
    return KokoroEngine


def _load_orpheus_engine():
    from .orpheus_engine import OrpheusEngine, OrpheusVoice
    globals()["OrpheusEngine"] = OrpheusEngine
    globals()["OrpheusVoice"] = OrpheusVoice
    return OrpheusEngine


def _load_zipvoice_engine():
    from .zipvoice_engine import ZipVoiceEngine, ZipVoiceVoice
    globals()["ZipVoiceEngine"] = ZipVoiceEngine
    globals()["ZipVoiceVoice"] = ZipVoiceVoice
    return ZipVoiceEngine


# Map attribute names to lazy loader functions.
_lazy_imports = {
    "AzureEngine": _load_azure_engine,
    "AzureVoice": _load_azure_engine,
    "SystemEngine": _load_system_engine,
    "SystemVoice": _load_system_engine,
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
    "ZipVoiceEngine": _load_zipvoice_engine,
    "ZipVoiceVoice": _load_zipvoice_engine,
}


def __getattr__(name):
    if name in _lazy_imports:
        return _lazy_imports[name]()
    raise AttributeError(f"module {__name__} has no attribute {name}")
