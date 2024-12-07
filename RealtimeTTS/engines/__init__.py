from .base_engine import BaseEngine  # noqa: F401

# Optional dependencies
try:
    from .azure_engine import AzureEngine, AzureVoice  # noqa: F401
    import azure.cognitiveservices.speech as tts
except ImportError:
    AzureEngine, AzureVoice, tts = None, None, None

try:
    from .system_engine import SystemEngine, SystemVoice  # noqa: F401
    import pyttsx3
except ImportError:
    SystemEngine, SystemVoice, pyttsx3 = None, None, None

try:
    from .elevenlabs_engine import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenlabsEngine, ElevenlabsVoice, ElevenLabs = None, None, None

try:
    from .coqui_engine import CoquiEngine, CoquiVoice  # noqa: F401
    from TTS.utils.manage import ModelManager
except ImportError:
    CoquiEngine, CoquiVoice, ModelManager = None, None, None

try:
    from .openai_engine import OpenAIEngine, OpenAIVoice  # noqa: F401
    from openai import OpenAI
except ImportError:
    OpenAIEngine, OpenAIVoice, OpenAI = None, None, None

try:
    from .gtts_engine import GTTSEngine, GTTSVoice  # noqa: F401
    from gtts import gTTS
except ImportError:
    GTTSEngine, GTTSVoice, gTTS = None, None, None

try:
    from .parler_engine import ParlerEngine, ParlerVoice  # noqa: F401
except ImportError as e:
    ParlerEngine, ParlerVoice = None, None

try:
    from .edge_engine import EdgeEngine, EdgeVoice  # noqa: F401
except ImportError as e:
    EdgeEngine, EdgeVoice = None, None

try:
    from .style_engine import StyleTTSEngine  # noqa: F401
except ImportError as e:
    StyleTTSEngine = None