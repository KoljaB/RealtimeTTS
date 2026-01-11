if __name__ == "__main__":
    print("Starting server")
    import logging

    # Enable or disable debug logging
    DEBUG_LOGGING = False

    if DEBUG_LOGGING:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


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
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from queue import Queue, Empty
import threading
import logging
import uvicorn
import asyncio
import base64
import wave
import io
import os

PORT = int(os.environ.get("TTS_FASTAPI_PORT", 8000))

SUPPORTED_ENGINES = [
    "azure",
    "openai",
    "elevenlabs",
    "system",
    # "coqui",  #multiple queries are not supported on coqui engine right now, comment coqui out for tests where you need server start often,
    "kokoro"
]

# START_ENGINE will be set to the first successfully initialized engine
START_ENGINE = None

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

play_text_to_speech_semaphore = threading.Semaphore(1)
engines = {}
voices = {}
current_engine = None
speaking_lock = threading.Lock()
tts_lock = threading.Lock()
gen_lock = threading.Lock()


class TTSRequestHandler:
    def __init__(self, engine):
        self.engine = engine
        self.audio_queue = Queue()
        self.stream = TextToAudioStream(
            engine, on_audio_stream_stop=self.on_audio_stream_stop, muted=True
        )
        self.speaking = False
        self.generation_complete = threading.Event()

    def on_audio_chunk(self, chunk):
        self.audio_queue.put(chunk)

    def on_audio_stream_stop(self):
        self.audio_queue.put(None)
        self.speaking = False
        self.generation_complete.set()

    def play_text_to_speech(self, text):
        self.speaking = True
        self.stream.feed(text)
        logging.debug(f"Playing audio for text: {text}")
        print(f'Synthesizing: "{text}"')
        self.stream.play_async(on_audio_chunk=self.on_audio_chunk, muted=True)

    def audio_chunk_generator(self, send_wave_headers):
        first_chunk = False
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    print("Terminating stream")
                    break
                if not first_chunk:
                    if send_wave_headers:
                        print("Sending wave header")
                        yield create_wave_header_for_engine(self.engine)
                    first_chunk = True
                yield chunk
        except Exception as e:
            print(f"Error during streaming: {str(e)}")


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
    if engine_name not in engines:
        print(f"Warning: Engine '{engine_name}' not available")
        return False
    
    if current_engine is None:
        current_engine = engines[engine_name]
    else:
        current_engine = engines[engine_name]

    if voices[engine_name]:
        engines[engine_name].set_voice(voices[engine_name][0].name)
    
    return True


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


@app.get("/tts")
async def tts(request: Request, text: str = Query(...)):
    with tts_lock:
        request_handler = TTSRequestHandler(current_engine)
        browser_request = is_browser_request(request)

        if play_text_to_speech_semaphore.acquire(blocking=False):
            try:
                threading.Thread(
                    target=request_handler.play_text_to_speech,
                    args=(text,),
                    daemon=True,
                ).start()
            finally:
                play_text_to_speech_semaphore.release()

        return StreamingResponse(
            request_handler.audio_chunk_generator(browser_request),
            media_type="audio/wav",
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket endpoint for TTS - supports multiple concurrent users"""
    
    # Check if system engine is active - it doesn't support WebSocket mode
    if current_engine and current_engine.engine_name == "system":
        await websocket.accept()
        await websocket.send_json({
            "error": {
                "message": "System engine (pyttsx3) does not support WebSocket mode. Please switch to OpenAI, Kokoro, Azure, or ElevenLabs engine.",
                "engineName": "system"
            }
        })
        await websocket.close()
        print("WebSocket connection rejected - system engine not supported")
        return
    
    await websocket.accept()
    print("WebSocket client connected")
    
    request_queue = asyncio.Queue()
    is_active = True
    
    async def generate_audio(text: str):
        """Generate audio for given text and return chunks"""
        # Create dedicated handler for this request
        handler = TTSRequestHandler(current_engine)
        
        # Start generation in background thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, handler.play_text_to_speech, text)
        
        # Collect all audio chunks
        chunks = []
        while True:
            try:
                chunk = handler.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                chunks.append(chunk)
            except Empty:
                if handler.generation_complete.is_set():
                    break
                await asyncio.sleep(0.01)
        
        return chunks
    
    async def process_requests():
        """Process queued requests one by one"""
        while is_active:
            try:
                text = await asyncio.wait_for(request_queue.get(), timeout=1.0)
                if text is None:
                    break
                
                print(f"Processing: '{text}'")
                
                # Generate audio
                audio_chunks = await generate_audio(text)
                
                # Send WAV header
                wave_header = create_wave_header_for_engine(current_engine)
                _, _, sample_rate = current_engine.get_stream_info()
                
                await websocket.send_json({
                    "audioOutput": {
                        "audio": base64.b64encode(wave_header).decode('utf-8'),
                        "format": "wav",
                        "sampleRate": sample_rate,
                        "isHeader": True
                    }
                })
                
                # Send audio chunks
                for chunk in audio_chunks:
                    await websocket.send_json({
                        "audioOutput": {
                            "audio": base64.b64encode(chunk).decode('utf-8')
                        }
                    })
                
                # Send completion signal
                await websocket.send_json({
                    "finalOutput": {
                        "isFinal": True
                    }
                })
                print(f"Sent {len(audio_chunks)} chunks")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing request: {e}")
                break
    
    async def receive_messages():
        """Receive and queue text messages"""
        nonlocal is_active
        try:
            while is_active:
                text = await websocket.receive_text()
                if text.strip() and text.strip().lower() != "stop":
                    await request_queue.put(text)
                    print(f"Queued: '{text}'")
                else:
                    is_active = False
                    await request_queue.put(None)
                    break
        except WebSocketDisconnect:
            print("Client disconnected")
            is_active = False
            await request_queue.put(None)
        except Exception as e:
            print(f"Error receiving: {e}")
            is_active = False
            await request_queue.put(None)
    
    try:
        await asyncio.gather(receive_messages(), process_requests())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        is_active = False
        try:
            await websocket.close()
        except:
            pass
        print("WebSocket closed")


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
                .mode-selector {{
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .mode-selector button {{
                    width: auto;
                    padding: 10px 20px;
                }}
                .mode-selector button.active {{
                    background-color: #28a745;
                }}
                .status {{
                    text-align: center;
                    margin: 10px 0;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                <h2>Text to Speech</h2>
                <div class="mode-selector">
                    <button id="httpMode" class="active">HTTP Mode</button>
                    <button id="wsMode">WebSocket Mode</button>
                </div>
                <div class="status" id="status">Mode: HTTP</div>
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
                <audio id="audio" controls></audio>
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
            else:
                print("Azure engine skipped - missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")

        if "elevenlabs" == engine_name:
            elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
            if elevenlabs_api_key:
                print("Initializing elevenlabs engine")
                engines["elevenlabs"] = ElevenlabsEngine(elevenlabs_api_key)
            else:
                print("Elevenlabs engine skipped - missing ELEVENLABS_API_KEY")

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
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                print("Initializing openai engine")
                engines["openai"] = OpenAIEngine()
            else:
                print("OpenAI engine skipped - missing OPENAI_API_KEY")

    for _engine in engines.keys():
        print(f"Retrieving voices for TTS Engine {_engine}")
        try:
            voices[_engine] = engines[_engine].get_voices()
        except Exception as e:
            voices[_engine] = []
            logging.error(f"Error retrieving voices for {_engine}: {str(e)}")

    # Set START_ENGINE to first available engine
    if not engines:
        print("Error: No TTS engines were successfully initialized!")
        print("Please check your API keys and configuration.")
        exit(1)
    
    # Try to use the first engine from SUPPORTED_ENGINES that was initialized
    START_ENGINE = None
    for engine_name in SUPPORTED_ENGINES:
        if engine_name in engines:
            START_ENGINE = engine_name
            break
    
    if START_ENGINE:
        _set_engine(START_ENGINE)
        print(f"Default engine set to: {START_ENGINE}")
    else:
        print("Error: Could not set default engine")
        exit(1)

    print("Server ready")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
