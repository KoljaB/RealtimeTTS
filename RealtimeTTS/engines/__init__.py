from .base_engine import BaseEngine  # noqa: F401

# Optional dependencies
try:
    from .azure_engine import AzureEngine, AzureVoice  # noqa: F401
    import azure.cognitiveservices.speech as tts
except ImportError:
    AzureEngine, AzureVoice = None, None

try:
    from .system_engine import SystemEngine, SystemVoice  # noqa: F401
    import pyttsx3
except ImportError:
    SystemEngine, SystemVoice = None, None

try:
    from .elevenlabs_engine import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenlabsEngine, ElevenlabsVoice = None, None

try:
    from .coqui_engine import CoquiEngine, CoquiVoice  # noqa: F401
    from TTS.utils.manage import ModelManager
except ImportError:
    CoquiEngine, CoquiVoice = None, None

try:
    from .openai_engine import OpenAIEngine, OpenAIVoice  # noqa: F401
    from openai import OpenAI
except ImportError:
    OpenAIEngine, OpenAIVoice = None, None

try:
    from .gtts_engine import GTTSEngine, GTTSVoice  # noqa: F401
    from gtts import gTTS
except ImportError:
    GTTSEngine, GTTSVoice = None, None
