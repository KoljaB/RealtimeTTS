from .text_to_stream import TextToAudioStream  # noqa: F401
from .engines import BaseEngine  # noqa: F401

# Optional dependencies
try:
    from .engines import SystemEngine, SystemVoice  # noqa: F401
    import pyttsx3
except ImportError:
    SystemEngine, SystemVoice, pyttsx3 = None, None, None

try:
    from .engines import AzureEngine, AzureVoice  # noqa: F401
    import azure.cognitiveservices.speech as tts
except ImportError:
    AzureEngine, AzureVoice, tts = None, None, None

try:
    from .engines import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenlabsEngine, ElevenlabsVoice, ElevenLabs = None, None, None

try:
    from .engines import CoquiEngine, CoquiVoice  # noqa: F401
    from TTS.utils.manage import ModelManager
except ImportError:
    CoquiEngine, CoquiVoice, ModelManager = None, None, None

try:
    from .engines import OpenAIEngine, OpenAIVoice  # noqa: F401
    from openai import OpenAI
except ImportError:
    OpenAIEngine, OpenAIVoice, OpenAI = None, None, None

try:
    from .engines import GTTSEngine, GTTSVoice  # noqa: F401
    from gtts import gTTS
except ImportError:
    GTTSEngine, GTTSVoice, gTTS = None, None, None

try:
    from .engines import ParlerEngine, ParlerVoice  # noqa: F401
except ImportError:
    ParlerEngine, ParlerVoice = None, None
