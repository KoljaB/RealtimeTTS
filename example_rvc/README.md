# XTTSv2 with RVC Post-processing

This project provides real-time text-to-speech capabilities using the Coqui XTTS model and employs real-time voice conversion (RVC) post-processing for optimizing the TTS output quality.

## Prerequisites

- Python 3.10.9
- Torch version 2.3.1+cu121 is recommended

To install the recommended Torch version:
```
pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the necessary models:
   ```
   python download_models.py
   ```

## Usage

The main class of this project is `XTTSRVCSynthesizer`. You can use it in your own projects or try the provided test scripts.

### XTTSRVCSynthesizer

The `XTTSRVCSynthesizer` class is initialized with the following parameters:

- `xtts_model` (str, optional): Path to the folder containing the XTTS files (model.pth, config.json, vocab.json, speakers_xtts.pth).
- `xtts_voice` (str, optional): Path to the file with the ~5-10 second mono 22050 Hz reference voice wave file.
- `rvc_model` (str, optional): Path to the .pth file with the RVC reference. The .index file should be in the same folder and have the same name as this .pth file.
- `rvc_sample_rate` (int, default=40000): The sample rate the RVC model was trained against (usually 40000 or 48000).
- `use_logging` (bool, default=False): Enable extended debug logging.

### Key Methods

- `push_text(text: str)`: Pushes text for synthesis. As soon as enough text is pushed, it gets automatically synthesized.
- `synthesize()`: Starts TTS immediately. This is a blocking method that waits until playout is finished.
- `load_xtts_model()`: Loads the XTTS model.
- `load_rvc_model()`: Loads the RVC model.

### Testing

You can test the synthesizer using the provided scripts:

1. Basic test:
   ```
   python xtts_rvc_tester.py
   ```

2. Test with LLM integration:
   ```
   python xtts_rvc_llm.py
   ```

## Project Structure

- `xtts_rvc_synthesizer.py`: Contains the main `XTTSRVCSynthesizer` class.
- `xtts_rvc_tester.py`: A script to test basic functionality of the synthesizer.
- `xtts_rvc_llm.py`: Demonstrates integration with an LLM (Large Language Model) for generating speech content.
- `download_models.py`: Script to download required model files from Hugging Face.
- `bufferstream.py`: Implements a buffer stream for managing text input.
- `requirements.txt`: Lists all Python dependencies.

## Dependencies

Main dependencies include:
- fairseq
- faiss-cpu
- tensorboardX
- torchcrepe
- torchfcpe
- praat-parselmouth
- pyworld

For a complete list, refer to `requirements.txt`.

## Note

Ensure you run `download_models.py` before attempting to use `xtts_rvc_tester.py` or `xtts_rvc_llm.py`. This script will download the necessary model files from Hugging Face.
