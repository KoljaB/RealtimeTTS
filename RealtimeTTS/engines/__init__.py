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
    "PocketTTSEngine", "PocketTTSVoice",
    "NeuTTSEngine", "NeuTTSVoice",
    "CambEngine", "CambVoice",
    "ModelsLabEngine", "ModelsLabVoice",
    "MiniMaxEngine", "MiniMaxVoice",
    "CartesiaEngine", "CartesiaVoice",
    "FasterQwenEngine", "FasterQwenVoice",
    "OmniVoiceEngine", "OmniVoiceVoice",
    "TypecastEngine", "TypecastVoice",
    "LuxTTSEngine", "LuxTTSVoice",
    "ChatterboxEngine", "ChatterboxVoice",
    "SoproTTSEngine", "SoproTTSVoice",
    "SopranoEngine", "SopranoVoice",
    "MossTTSEngine", "MossTTSVoice",
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


def _load_camb_engine():
    from .camb_engine import CambEngine, CambVoice
    globals()["CambEngine"] = CambEngine
    globals()["CambVoice"] = CambVoice
    return CambEngine


def _load_modelslab_engine():
    from .modelslab_engine import ModelsLabEngine, ModelsLabVoice
    globals()["ModelsLabEngine"] = ModelsLabEngine
    globals()["ModelsLabVoice"] = ModelsLabVoice
    return ModelsLabEngine


def _load_minimax_engine():
    from .minimax_engine import MiniMaxEngine, MiniMaxVoice
    globals()["MiniMaxEngine"] = MiniMaxEngine
    globals()["MiniMaxVoice"] = MiniMaxVoice
    return MiniMaxEngine


def _load_pocket_engine():
    from .pocket_engine import PocketTTSEngine, PocketTTSVoice
    globals()["PocketTTSEngine"] = PocketTTSEngine
    globals()["PocketTTSVoice"] = PocketTTSVoice
    return PocketTTSEngine


def _load_neutts_engine():
    from .neutts_engine import NeuTTSEngine, NeuTTSVoice
    globals()["NeuTTSEngine"] = NeuTTSEngine
    globals()["NeuTTSVoice"] = NeuTTSVoice
    return NeuTTSEngine


def _load_cartesia_engine():
    from .cartesia_engine import CartesiaEngine, CartesiaVoice
    globals()["CartesiaEngine"] = CartesiaEngine
    globals()["CartesiaVoice"] = CartesiaVoice
    return CartesiaEngine


def _load_fasterqwen_engine():
    from .faster_qwen_engine import FasterQwenEngine, FasterQwenVoice
    globals()["FasterQwenEngine"] = FasterQwenEngine
    globals()["FasterQwenVoice"] = FasterQwenVoice
    return FasterQwenEngine


def _load_omni_voice_engine():
    from .omnivoice_engine import OmniVoiceEngine, OmniVoiceVoice
    globals()["OmniVoiceEngine"] = OmniVoiceEngine
    globals()["OmniVoiceVoice"] = OmniVoiceVoice
    return OmniVoiceEngine


def _load_typecast_engine():
    from .typecast_engine import TypecastEngine, TypecastVoice
    globals()["TypecastEngine"] = TypecastEngine
    globals()["TypecastVoice"] = TypecastVoice
    return TypecastEngine


def _load_luxtts_engine():
    from .luxtts_engine import LuxTTSEngine, LuxTTSVoice
    globals()["LuxTTSEngine"] = LuxTTSEngine
    globals()["LuxTTSVoice"] = LuxTTSVoice
    return LuxTTSEngine


def _load_chatterbox_engine():
    from .chatterbox_engine import ChatterboxEngine, ChatterboxVoice
    globals()["ChatterboxEngine"] = ChatterboxEngine
    globals()["ChatterboxVoice"] = ChatterboxVoice
    return ChatterboxEngine


def _load_sopro_engine():
    from .sopro_engine import SoproTTSEngine, SoproTTSVoice
    globals()["SoproTTSEngine"] = SoproTTSEngine
    globals()["SoproTTSVoice"] = SoproTTSVoice
    return SoproTTSEngine


def _load_soprano_engine():
    from .soprano_engine import SopranoEngine, SopranoVoice
    globals()["SopranoEngine"] = SopranoEngine
    globals()["SopranoVoice"] = SopranoVoice
    return SopranoEngine


def _load_moss_tts_engine():
    from .moss_tts_engine import MossTTSEngine, MossTTSVoice
    globals()["MossTTSEngine"] = MossTTSEngine
    globals()["MossTTSVoice"] = MossTTSVoice
    return MossTTSEngine

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
    "PocketTTSEngine": _load_pocket_engine,
    "PocketTTSVoice": _load_pocket_engine,
    "NeuTTSEngine": _load_neutts_engine,
    "NeuTTSVoice": _load_neutts_engine,
    "CambEngine": _load_camb_engine,
    "CambVoice": _load_camb_engine,
    "ModelsLabEngine": _load_modelslab_engine,
    "ModelsLabVoice": _load_modelslab_engine,
    "MiniMaxEngine": _load_minimax_engine,
    "MiniMaxVoice": _load_minimax_engine,
    "CartesiaEngine": _load_cartesia_engine,
    "CartesiaVoice": _load_cartesia_engine,
    "FasterQwenEngine": _load_fasterqwen_engine,
    "FasterQwenVoice": _load_fasterqwen_engine,
    "OmniVoiceEngine": _load_omni_voice_engine,
    "OmniVoiceVoice": _load_omni_voice_engine,
    "TypecastEngine": _load_typecast_engine,
    "TypecastVoice": _load_typecast_engine,
    "LuxTTSEngine": _load_luxtts_engine,
    "LuxTTSVoice": _load_luxtts_engine,
    "ChatterboxEngine": _load_chatterbox_engine,
    "ChatterboxVoice": _load_chatterbox_engine,
    "SoproTTSEngine": _load_sopro_engine,
    "SoproTTSVoice": _load_sopro_engine,
    "SopranoEngine": _load_soprano_engine,
    "SopranoVoice": _load_soprano_engine,
    "MossTTSEngine": _load_moss_tts_engine,
    "MossTTSVoice": _load_moss_tts_engine,
}


def __getattr__(name):
    if name in _lazy_imports:
        return _lazy_imports[name]()
    raise AttributeError(f"module {__name__} has no attribute {name}")
