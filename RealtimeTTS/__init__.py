from .text_to_stream import TextToAudioStream  # noqa: F401
from .engines import BaseEngine  # noqa: F401

# Optional dependencies
try:
    from .engines import SystemEngine, SystemVoice  # noqa: F401
except ImportError:
    SystemEngine, SystemVoice, pyttsx3 = None, None, None

try:
    from .engines import AzureEngine, AzureVoice  # noqa: F401
except ImportError:
    AzureEngine, AzureVoice, tts = None, None, None

try:
    from .engines import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
except ImportError:
    ElevenlabsEngine, ElevenlabsVoice, ElevenLabs = None, None, None

try:
    from .engines import CoquiEngine, CoquiVoice  # noqa: F401
except ImportError:
    CoquiEngine, CoquiVoice, ModelManager = None, None, None

try:
    from .engines import OpenAIEngine, OpenAIVoice  # noqa: F401
except ImportError:
    OpenAIEngine, OpenAIVoice, OpenAI = None, None, None

try:
    from .engines import GTTSEngine, GTTSVoice  # noqa: F401
except ImportError:
    GTTSEngine, GTTSVoice, gTTS = None, None, None

try:
    from .engines import ParlerEngine, ParlerVoice  # noqa: F401
except ImportError:
    ParlerEngine, ParlerVoice = None, None

try:
    from .engines import EdgeEngine, EdgeVoice  # noqa: F401
except ImportError:
    EdgeEngine, EdgeVoice = None, None

try:
    from .engines import StyleTTSEngine, StyleTTSVoice  # noqa: F401
except ImportError:
    StyleTTSEngine, StyleTTSVoice = None

try:
    from .engines import PiperEngine, PiperVoice  # noqa: F401
except ImportError as e:
    PiperEngine, PiperVoice = None

try:
    from .engines import KokoroEngine  # noqa: F401
except ImportError as e:
    KokoroEngine = None

try:
    from .engines import OrpheusEngine, OrpheusVoice  # noqa: F401
except ImportError as e:
    OrpheusEngine, OrpheusVoice = None
