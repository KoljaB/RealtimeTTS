# Core module imports
from .text_to_stream import TextToAudioStream  # noqa: F401
from .engines import BaseEngine  # noqa: F401

# Optional dependencies for different text-to-speech engines
# Each section tries to import a specific engine and assigns None if unavailable.

# System TTS Engine (e.g., pyttsx3 for local system speech synthesis)
try:
    from .engines import SystemEngine, SystemVoice  # noqa: F401
    import pyttsx3
except ImportError:
    # Assign None if pyttsx3 is not installed
    SystemEngine = SystemVoice = pyttsx3 = None

# Azure TTS Engine (uses Microsoft Azure Cognitive Services)
try:
    from .engines import AzureEngine, AzureVoice  # noqa: F401
    import azure.cognitiveservices.speech as tts
except ImportError:
    # Assign None if Azure TTS is not available
    AzureEngine = AzureVoice = tts = None

# ElevenLabs TTS Engine (uses Eleven Labs' API for TTS services)
try:
    from .engines import ElevenlabsEngine, ElevenlabsVoice  # noqa: F401
    from elevenlabs.client import ElevenLabs
except ImportError:
    # Assign None if Eleven Labs' client is not installed
    ElevenlabsEngine = ElevenlabsVoice = ElevenLabs = None

# Coqui TTS Engine (open-source TTS with neural network models)
try:
    from .engines import CoquiEngine, CoquiVoice  # noqa: F401
    from TTS.utils.manage import ModelManager
except ImportError:
    # Assign None if Coqui TTS is not available
    CoquiEngine = CoquiVoice = ModelManager = None

# OpenAI TTS Engine (uses OpenAI's API, if available)
try:
    from .engines import OpenAIEngine, OpenAIVoice  # noqa: F401
    from openai import OpenAI
except ImportError:
    # Assign None if OpenAI API is not installed
    OpenAIEngine = OpenAIVoice = OpenAI = None

# Google TTS Engine (gTTS for Google Text-to-Speech)
try:
    from .engines import GTTSEngine, GTTSVoice  # noqa: F401
    from gtts import gTTS
except ImportError:
    # Assign None if gTTS is not installed
    GTTSEngine = GTTSVoice = gTTS = None

# Function to check the availability of TTS engines
def get_available_engines():
    """Returns a dictionary of available TTS engines with their status."""
    return {
        "SystemEngine": SystemEngine is not None,
        "AzureEngine": AzureEngine is not None,
        "ElevenlabsEngine": ElevenlabsEngine is not None,
        "CoquiEngine": CoquiEngine is not None,
        "OpenAIEngine": OpenAIEngine is not None,
        "GTTSEngine": GTTSEngine is not None
    }

# Function to automatically select an available TTS engine based on preference
def select_default_engine(preferred_engines=None):
    """
    Selects the default engine based on user preference or availability.
    
    Args:
        preferred_engines (list): Ordered list of engine names to prioritize.
        
    Returns:
        The first available engine from the preferred list, or None if none are available.
    """
    available_engines = get_available_engines()
    if preferred_engines is None:
        preferred_engines = ["SystemEngine", "AzureEngine", "ElevenlabsEngine", "CoquiEngine", "OpenAIEngine", "GTTSEngine"]
    
    for engine_name in preferred_engines:
        if available_engines.get(engine_name):
            return globals()[engine_name]  # Returns the engine module itself
    return None

# Logging for import status
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the import status for each engine
for engine_name, status in get_available_engines().items():
    if status:
        logger.info(f"{engine_name} is available.")
    else:
        logger.warning(f"{engine_name} is not available. Ensure dependencies are installed.")

# Function to handle TTS conversion with error handling for unsupported features
def use_engine_for_tts(engine, text):
    """
    Utilizes the specified engine to perform TTS. Handles errors if the engine is unavailable or unsupported.
    
    Args:
        engine: The TTS engine instance.
        text (str): Text to convert to speech.
    """
    if engine is None:
        raise ValueError("Requested engine is unavailable. Please check dependencies or select another engine.")
    
    try:
        # Call the TTS function for the given engine
        engine.speak(text)  # Hypothetical method; replace with actual call
    except AttributeError:
        logger.error("Selected engine does not support TTS functionality.")
    except Exception as e:
        logger.error(f"An error occurred during TTS conversion: {e}")

# Function to set customizable TTS output settings
def set_engine_settings(engine, language="en-US", pitch=1.0, speed=1.0):
    """
    Sets TTS engine settings for language, pitch, and speed where supported.
    
    Args:
        engine: The TTS engine instance.
        language (str): Language code (e.g., "en-US").
        pitch (float): Pitch adjustment (1.0 is default).
        speed (float): Speed adjustment (1.0 is default).
    """
    if engine is None:
        raise ValueError("Engine is not available.")
    
    # Hypothetical properties; adjust based on actual engine support
    if hasattr(engine, "set_language"):
        engine.set_language(language)
    if hasattr(engine, "set_pitch"):
        engine.set_pitch(pitch)
    if hasattr(engine, "set_speed"):
        engine.set_speed(speed)
