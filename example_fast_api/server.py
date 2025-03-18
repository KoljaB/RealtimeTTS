import argparse
import os

parser = argparse.ArgumentParser(description="Run the TTS FastAPI server.")
parser.add_argument("-p", "--port", type=int, default=int(os.environ.get("TTS_FASTAPI_PORT", 8000)),
                    help="Port to run the FastAPI server on (default: 8000 or TTS_FASTAPI_PORT env var).")
parser.add_argument('-D', '--debug', action='store_true', help='Enable debug logging for detailed server operations')

args = parser.parse_args()

PORT = args.port
DEBUG_LOGGING = args.debug

if __name__ == "__main__":
    import logging

    if DEBUG_LOGGING:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")

from RealtimeTTS import (
    TextToAudioStream,
    AzureEngine,
    ElevenlabsEngine,
    SystemEngine,
    CoquiEngine,
    OpenAIEngine,
    KokoroEngine
)

from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.staticfiles import StaticFiles

from queue import Queue
import threading
import logging
import uvicorn
import wave
import io

SUPPORTED_ENGINES = [
    "azure",
    "openai",
    "elevenlabs",
    "system",
    "coqui",  # comment coqui out for tests where you need server start often,
    "kokoro"
]

# change start engine:
START_ENGINE = SUPPORTED_ENGINES[0]

BROWSER_IDENTIFIERS = [
    "mozilla",
    "chrome",
    "safari",
    "firefox",
    "edge",
    "opera",
    "msie",
    "trident",
]

origins = [
    "http://localhost",
    f"http://localhost:{PORT}",
    "http://127.0.0.1",
    f"http://127.0.0.1:{PORT}",
    "https://localhost",
    f"https://localhost:{PORT}",
    "https://127.0.0.1",
    f"https://127.0.0.1:{PORT}",
]

audio_queue = Queue()
play_text_to_speech_semaphore = threading.Semaphore(1)
engines = {}
voices = {}
current_engine = None
stream = None
current_speaking = {}
speaking_lock = threading.Lock()
tts_lock = threading.Lock()
gen_lock = threading.Lock()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a CSP that allows 'self' for script sources for firefox
csp = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self' 'unsafe-inline'",
    "img-src": "'self' data:",
    "font-src": "'self' data:",
    "media-src": "'self' blob:",
}
csp_string = "; ".join(f"{key} {value}" for key, value in csp.items())


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = csp_string
    return response


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


def _set_engine(engine_name):
    global current_engine, stream
    if current_engine is None:
        current_engine = engines[engine_name]
        stream = TextToAudioStream(current_engine, muted=True)
    else:
        current_engine = engines[engine_name]
        stream.load_engine(current_engine)

    if voices[engine_name]:
        engines[engine_name].set_voice(voices[engine_name][0].name)


@app.get("/set_engine")
def set_engine(request: Request, engine_name: str = Query(...)):
    if engine_name not in engines:
        return {"error": "Engine not supported"}

    try:
        _set_engine(engine_name)
        return {"message": f"Switched to {engine_name} engine"}
    except Exception as e:
        logging.error(f"Error switching engine: {str(e)}")
        return {"error": "Failed to switch engine"}


def play_text_to_speech(stream, text, audio_queue):
    set_speaking(text, True)

    def on_audio_chunk(chunk):
        logging.debug("Received chunk")
        audio_queue.put(chunk)

    try:
        stream.feed(text)
        logging.debug(f"Playing audio for text: {text}")
        print(f'Synthesizing: "{text}"')
        stream.play(on_audio_chunk=on_audio_chunk, muted=True)
        audio_queue.put(None)
    finally:
        set_speaking(text, False)
        play_text_to_speech_semaphore.release()


def is_browser_request(request):
    user_agent = request.headers.get("user-agent", "").lower()
    is_browser = any(browser_id in user_agent for browser_id in BROWSER_IDENTIFIERS)
    return is_browser


def create_wave_header_for_engine(engine):
    _, _, sample_rate = engine.get_stream_info()

    num_channels = 1
    sample_width = 2
    frame_rate = sample_rate

    wav_header = io.BytesIO()
    with wave.open(wav_header, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)

    wav_header.seek(0)
    wave_header_bytes = wav_header.read()
    wav_header.close()

    # Create a new BytesIO with the correct MIME type for Firefox
    final_wave_header = io.BytesIO()
    final_wave_header.write(wave_header_bytes)
    final_wave_header.seek(0)

    return final_wave_header.getvalue()

def audio_chunk_generator(audio_queue, send_wave_headers):
    with gen_lock:
        first_chunk = False
        try:
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    logging.debug("Terminating stream")
                    break
                if not first_chunk:
                    if (
                        send_wave_headers
                        and not current_engine.engine_name == "elevenlabs"
                    ):
                        logging.debug("Sending wave header")
                        yield create_wave_header_for_engine(current_engine)
                    first_chunk = True
                logging.debug("Sending chunk")
                yield chunk
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")


def is_currently_speaking(text):
    with speaking_lock:
        return current_speaking.get(text, False)


def set_speaking(text, status):
    with speaking_lock:
        current_speaking[text] = status


@app.get("/tts")
def tts(request: Request, text: str = Query(...)):
    browser_request = is_browser_request(request)
    audio_queue = Queue()

    if play_text_to_speech_semaphore.acquire(blocking=False):
        threading.Thread(
            target=play_text_to_speech, args=(stream, text, audio_queue), daemon=True
        ).start()
    else:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable, currently processing another request. Please try again shortly.",
            headers={"Retry-After": "10"},
        )

    return StreamingResponse(
        audio_chunk_generator(audio_queue, browser_request),
        media_type="audio/wav"
        if current_engine.engine_name != "elevenlabs"
        else "audio/mpeg",
    )


@app.get("/tts-text")
def tts_text(request: Request, text: str = Query(...)):
    if "favicon.ico" in request.url.path:
        print("favicon requested")
        return FileResponse("static/favicon.ico")

    print(f"/tts_text route synthesizing text: {text}")

    browser_request = is_browser_request(request)

    if play_text_to_speech_semaphore.acquire(blocking=False):
        threading.Thread(
            target=play_text_to_speech, args=(stream, text), daemon=True
        ).start()
    else:
        logging.debug("Can't play audio, another instance is already running")

    return StreamingResponse(
        audio_chunk_generator(browser_request), media_type="audio/wav"
    )


@app.get("/engines")
def get_engines():
    return list(engines.keys())


@app.get("/voices")
def get_voices():
    voices_list = []
    for voice in voices[current_engine.engine_name]:
        voices_list.append(voice.name)
    return voices_list


@app.get("/setvoice")
def set_voice(request: Request, voice_name: str = Query(...)):
    print(f"Getting request: {voice_name}")
    if not current_engine:
        print("No engine is currently selected")
        return {"error": "No engine is currently selected"}

    try:
        print(f"Setting voice to {voice_name}")
        current_engine.set_voice(voice_name)
        return {"message": f"Voice set to {voice_name} successfully"}
    except Exception as e:
        print(f"Error setting voice: {str(e)}")
        logging.error(f"Error setting voice: {str(e)}")
        return {"error": "Failed to set voice"}


@app.get("/")
def root_page():
    engines_options = "".join(
        [
            f'<option value="{engine}">{engine.title()}</option>'
            for engine in engines.keys()
        ]
    )
    content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text-To-Speech</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    margin: 0;
                    padding: 0;
                }}
                h2 {{
                    color: #333;
                    text-align: center;
                }}
                #container {{
                    width: 80%;
                    margin: 50px auto;
                    background-color: #fff;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                label {{
                    font-weight: bold;
                }}
                select, textarea {{
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-sizing: border-box;
                    font-size: 16px;
                }}
                button {{
                    display: block;
                    width: 100%;
                    padding: 15px;
                    background-color: #007bff;
                    border: none;
                    border-radius: 5px;
                    color: #fff;
                    font-size: 16px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }}
                button:hover {{
                    background-color: #0056b3;
                }}
                audio {{
                    width: 80%;
                    margin: 10px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                <h2>Text to Speech</h2>
                <label for="engine">Select Engine:</label>
                <select id="engine">
                    {engines_options}
                </select>
                <label for="voice">Select Voice:</label>
                <select id="voice">
                    <!-- Options will be dynamically populated by JavaScript -->
                </select>
                <textarea id="text" rows="4" cols="50" placeholder="Enter text here..."></textarea>
                <button id="speakButton">Speak</button>
                <audio id="audio" controls></audio> <!-- Hidden audio player -->
            </div>
            <script src="/static/tts.js"></script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    print("Initializing TTS Engines")

    for engine_name in SUPPORTED_ENGINES:
        if "azure" == engine_name:
            azure_api_key = os.environ.get("AZURE_SPEECH_KEY")
            azure_region = os.environ.get("AZURE_SPEECH_REGION")
            if azure_api_key and azure_region:
                print("Initializing azure engine")
                engines["azure"] = AzureEngine(azure_api_key, azure_region)

        if "elevenlabs" == engine_name:
            elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
            if elevenlabs_api_key:
                print("Initializing elevenlabs engine")
                engines["elevenlabs"] = ElevenlabsEngine(elevenlabs_api_key)

        if "system" == engine_name:
            print("Initializing system engine")
            engines["system"] = SystemEngine()

        if "coqui" == engine_name:
            print("Initializing coqui engine")
            engines["coqui"] = CoquiEngine()

        if "kokoro" == engine_name:
            print("Initializing kokoro engine")
            engines["kokoro"] = KokoroEngine()

        if "openai" == engine_name:
            print("Initializing openai engine")
            engines["openai"] = OpenAIEngine()


    for _engine in engines.keys():
        print(f"Retrieving voices for TTS Engine {_engine}")
        try:
            voices[_engine] = engines[_engine].get_voices()
        except Exception as e:
            voices[_engine] = []
            logging.error(f"Error retrieving voices for {_engine}: {str(e)}")

    _set_engine(START_ENGINE)

    print("Server ready")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
