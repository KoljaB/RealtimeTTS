import requests
import pyaudio

text_to_tts = "Hello World"

# Configuration
SERVER_URL = "http://localhost:8000/tts"
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # coqui (24000), azure (16000), openai (22050), system (22050)
CHUNK_SIZE = 1024

# Initialize PyAudio
pyaudio_instance = pyaudio.PyAudio()
stream = pyaudio_instance.open(
    format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, output=True
)


# Function to play audio stream
def play_audio(response):
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            stream.write(chunk)


# Function to request text-to-speech conversion
def request_tts(text):
    response = requests.get(SERVER_URL, params={"text": text}, stream=True)
    play_audio(response)


# Test the client
try:
    request_tts(text_to_tts)
finally:
    stream.stop_stream()
    stream.close()
    pyaudio_instance.terminate()
