from .text_to_stream import TextToAudioStream  # noqa: F401
from .engines import BaseEngine  # noqa: F401

# Optional dependencies
try:
    from .engines import SystemEngine, SystemVoice  # noqa: F401
    import pyttsx3
except ImportError:
    SystemEngine, SystemVoice = None, None

try:
    from .engines import AzureEngine, AzureVoice  # noqa: F401
    import azure.cognitiveservices.speech as tts
except ImportError:
    AzureEngine, AzureVoice = None, None

try:
    from .engines import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenlabsEngine, ElevenlabsVoice = None, None

try:
    from .engines import CoquiEngine, CoquiVoice  # noqa: F401
    from TTS.utils.manage import ModelManager
except ImportError:
    CoquiEngine, CoquiVoice = None, None

try:
    from .engines import OpenAIEngine, OpenAIVoice  # noqa: F401
    from openai import OpenAI
except ImportError:
    OpenAIEngine, OpenAIVoice = None, None

try:
    from .engines import GTTSEngine, GTTSVoice  # noqa: F401
    from gtts import gTTS
except ImportError:
    GTTSEngine, GTTSVoice = None, None

# from .text_to_stream import TextToAudioStream  # noqa: F401
# from .engines import BaseEngine  # noqa: F401
# from .engines import SystemEngine, SystemVoice  # noqa: F401
# from .engines import AzureEngine, AzureVoice  # noqa: F401
# from .engines import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
# from .engines import CoquiEngine, CoquiVoice  # noqa: F401
# from .engines import OpenAIEngine, OpenAIVoice  # noqa: F401
# from .engines import GTTSEngine, GTTSVoice
