import requests
import pyaudio
import time
import wave
import threading
import argparse
from queue import Queue

# Argument parser setup
parser = argparse.ArgumentParser(description="Run the TTS client.")
parser.add_argument("-p", "--port", type=int, default=8000, help="Port of the TTS server (default: 8000)")
parser.add_argument(
    "-t",
    "--text",
    type=str,
    default="Hello! This is a default text to speech demo text!",
    help="Text to convert to speech (default: 'Hello! This is a default text-to-speech demonstration. Enjoy the experience!')"
)
parser.add_argument(
    "-w",
    "--write",
    action="store_true",
    help="Save output to a WAV file"
)
args = parser.parse_args()

port = args.port
text_to_tts = args.text
write_to_file = args.write

# Configuration

SERVER_URL = f"http://127.0.0.1:{port}/tts"
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # coqui (24000), azure (16000), openai (22050), system (22050)

# Initialize PyAudio
pyaudio_instance = pyaudio.PyAudio()
stream = pyaudio_instance.open(
    format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, output=True
)


# Optionally set up WAV file
if write_to_file:
    output_wav_file = 'output_audio.wav'
    wav_file = wave.open(output_wav_file, 'wb')
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(pyaudio_instance.get_sample_size(AUDIO_FORMAT))
    wav_file.setframerate(RATE)

# Queue for chunk storage
chunk_queue = Queue()

# Thread function for playing audio
def play_audio():
    global start_time
    buffer = b""
    played_out = False
    got_first_chunk = False

    frame_size = pyaudio_instance.get_sample_size(AUDIO_FORMAT) * CHANNELS
    min_buffer_size = 1024 * 6   # Adjust as needed

    # Initial buffering
    while len(buffer) < min_buffer_size:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        if not got_first_chunk:
            got_first_chunk = True
            time_to_first_token = time.time() - start_time
            print(f"Time to first token: {time_to_first_token}")
        buffer += chunk

    # Now start playback
    while True:
        # Write data if buffer has enough frames
        if len(buffer) >= frame_size:
            num_frames = len(buffer) // frame_size
            bytes_to_write = num_frames * frame_size
            if not played_out:
                played_out = True
                time_to_first_token = time.time() - start_time
                # print(f"Time to first playout: {time_to_first_token}")
            stream.write(buffer[:bytes_to_write])
            buffer = buffer[bytes_to_write:]
        else:
            # Get more data
            chunk = chunk_queue.get()
            if chunk is None:
                # Write any remaining data
                if len(buffer) > 0:
                    # Truncate buffer to multiple of frame size if necessary
                    if len(buffer) % frame_size != 0:
                        buffer = buffer[:-(len(buffer) % frame_size)]

                    if not played_out:
                        played_out = True
                        time_to_first_token = time.time() - start_time
                        # print(f"Time to first playout: {time_to_first_token}")
                    stream.write(buffer)
                break
            buffer += chunk


# Function to request text-to-speech conversion and retrieve chunks
def request_tts(text):
    global start_time
    start_time = time.time()
    try:
        response = requests.get(SERVER_URL, params={"text": text}, stream=True, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if the response status is 4xx/5xx

        # Read data as it becomes available
        for chunk in response.iter_content(chunk_size=None):
            if chunk and write_to_file:
                wav_file.writeframes(chunk)
                chunk_queue.put(chunk)
        
        # Signal the end of the stream
        chunk_queue.put(None)

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        chunk_queue.put(None)  # Ensure that playback thread exits gracefully

# Start audio playback thread
playback_thread = threading.Thread(target=play_audio)
playback_thread.start()

# Retrieve and queue chunks in the main thread
try:
    request_tts(text_to_tts)
finally:
    playback_thread.join()
    stream.stop_stream()
    stream.close()
    pyaudio_instance.terminate()
