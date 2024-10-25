import requests
import pyaudio
import time

text_to_tts = "Hello World"

# Configuration
SERVER_URL = "http://127.0.0.1:8000/tts"
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # coqui (24000), azure (16000), openai (22050), system (22050)
CHUNK_SIZE = 1024

start_time = 0
first_chunk = False

# Initialize PyAudio
pyaudio_instance = pyaudio.PyAudio()
stream = pyaudio_instance.open(
    format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, output=True
)


# Function to play audio stream
def play_audio(response):
    global first_chunk
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            if not first_chunk:
                time_to_first_token = time.time() - start_time
                print(f"Time to first token: {time_to_first_token}")

            first_chunk = True
            stream.write(chunk)


# Function to request text-to-speech conversion
def request_tts(text):
    response = requests.get(SERVER_URL, params={"text": text}, stream=True)
    play_audio(response)


# Test the client
try:
    start_time = time.time()
    request_tts(text_to_tts)
finally:
    stream.stop_stream()
    stream.close()
    pyaudio_instance.terminate()
