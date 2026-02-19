# RealtimeTTS-ModelsLab

A [ModelsLab](https://modelslab.com/) TTS engine for [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) - a low-latency, real-time text-to-speech library for Python.

## Features

- 🚀 **Low-latency** text-to-speech synthesis
- 🌐 **Multiple languages** support (14+ languages)
- 🎤 **Multiple voices** (male and female voices)
- ⚡ **Adjustable speed** for speech rate
- 🔄 **Streaming support** for real-time playback
- 🌍 **Environment variable** configuration support

## Installation

```bash
pip install realtimetts-modelslab
```

Or install from source:

```bash
cd realtimetts-modelslab
pip install .
```

### Requirements

- Python 3.8+
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) >= 0.4.0
- requests >= 2.28.0

## Quick Start

```python
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.modelslab_engine import ModelsLabEngine

# Create engine with your API key
engine = ModelsLabEngine(
    api_key="YOUR_MODELSLAB_API_KEY",
    voice="male1",
    language="english"
)

# Create stream and play
stream = TextToAudioStream(engine)
stream.play("Hello, this is a test of the ModelsLab TTS engine!")
```

### Using Environment Variables

```python
import os
os.environ["MODELSLAB_API_KEY"] = "YOUR_API_KEY"

from RealtimeTTS import TextToAudioStream
from RealtimeTTS.modelslab_engine import create_engine_from_env

# Create engine from environment
engine = create_engine_from_env()

stream = TextToAudioStream(engine)
stream.play("Hello from environment-configured engine!")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your ModelsLab API key |
| `voice` | str | `"male1"` | Voice ID (male1, male2, male3, female1, female2, female3) |
| `language` | str | `"english"` | Language (english, german, french, spanish, etc.) |
| `speed` | float | `1.0` | Speech speed (0.5 - 2.0) |
| `output_format` | str | `"mp3"` | Output format |
| `emotion` | bool | `False` | Enable emotion in speech |
| `temp` | bool | `False` | Enable temperature control |

### Available Voices

- `male1`, `male2`, `male3`
- `female1`, `female2`, `female3`

### Available Languages

`english`, `german`, `french`, `spanish`, `italian`, `portuguese`, `polish`, `russian`, `dutch`, `arabic`, `chinese`, `japanese`, `korean`, `hindi`

## API Reference

### ModelsLabEngine

```python
from RealtimeTTS.modelslab_engine import ModelsLabEngine

engine = ModelsLabEngine(
    api_key="YOUR_API_KEY",
    voice="male1",
    language="english",
    speed=1.0
)
```

#### Methods

- `synthesize(text)` - Synthesize speech from text (blocking)
- `synthesize_streaming(text, callback)` - Stream synthesis
- `play(text_generator, ...)` - Play text as speech
- `play_async(text_generator, ...)` - Play asynchronously
- `stop()` - Stop playback
- `is_available()` - Check if engine is ready
- `get_voices()` - Get available voices
- `get_languages()` - Get supported languages

### create_engine_from_env()

Convenience function to create engine from environment variables:

```python
from RealtimeTTS.modelslab_engine import create_engine_from_env

# Requires MODELSLAB_API_KEY environment variable
engine = create_engine_from_env()
```

## Example: Text-to-Speech with Callback

```python
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.modelslab_engine import ModelsLabEngine

def on_audio_chunk(chunk):
    """Handle each audio chunk."""
    print(f"Received {len(chunk)} bytes of audio")

engine = ModelsLabEngine(
    api_key="YOUR_API_KEY",
    voice="female1",
    language="english"
)

stream = TextToAudioStream(engine)
engine.synthesize_streaming("Hello world!", on_audio_chunk)
```

## Get Your API Key

1. Visit [modelslab.com](https://modelslab.com/)
2. Sign up for an account
3. Go to Dashboard → API Keys
4. Copy your API key

## Documentation

- [ModelsLab TTS API Documentation](https://docs.modelslab.com/voice-cloning/text-to-speech)
- [RealtimeTTS Documentation](https://koljab.github.io/RealtimeTTS/en/)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) - The base library
- [ModelsLab](https://modelslab.com/) - TTS API provider
