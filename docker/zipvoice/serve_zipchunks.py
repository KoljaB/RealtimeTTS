"""
A minimal FastAPI server for real-time Text-to-Speech using ZipVoice.

This server exposes a single `/tts` endpoint that accepts a JSON request
specifying a voice and text, and streams the resulting raw audio chunks
back to the client as quickly as possible.

Features:
-   Uses RealtimeTTS with the ZipVoiceEngine.
-   Loads two pre-configured voices at startup.
-   Handles one TTS request at a time to prevent resource contention.
-   Streams audio back for low-latency responses.
"""
from __future__ import annotations

import os
import queue
import threading
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from RealtimeTTS import TextToAudioStream, ZipVoiceEngine, ZipVoiceVoice

# --- Configuration ---
ZIPVOICE_PROJECT_ROOT = os.getenv("ZIPVOICE_PROJECT_ROOT", "/opt/app-root/src/ZipVoice")

# IMPORTANT: Replace these with the actual paths and transcriptions for your voice prompts.
VOICE_ALPHA_WAV_PATH = os.getenv("VOICE_ALPHA_WAV_PATH", "reference1.wav")
VOICE_ALPHA_PROMPT_TEXT = os.getenv(
    "VOICE_ALPHA_PROMPT_TEXT",
    "Hi there! I'm really excited to try this out! I hope the speech sounds natural and warm - that's exactly what I'm going for!"
)
VOICE_BETA_WAV_PATH = os.getenv("VOICE_BETA_WAV_PATH", "reference2.wav")
VOICE_BETA_PROMPT_TEXT = os.getenv(
    "VOICE_BETA_PROMPT_TEXT",
    "Your voice just got supercharged! Crystal clear audio that flows like silk and hits like thunder!"
)

# --- Global State ---
# These will be initialized during the application lifespan startup.
engine: ZipVoiceEngine | None = None
stream: TextToAudioStream | None = None
AVAILABLE_VOICES: Dict[str, ZipVoiceVoice] = {}

# A semaphore to ensure only one TTS synthesis runs at a time, preventing GPU overload.
tts_semaphore = threading.Semaphore(1)


# --- Pydantic Model for the Request Body ---
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice: str = Field(..., min_length=1, description="The name of the voice to use (e.g., 'alpha-warm').")


# --- Core TTS Processing Logic (runs in a thread) ---
def process_tts_request(
    text_to_speak: str,
    selected_voice: ZipVoiceVoice,
    audio_queue: queue.Queue
):
    """
    Handles the actual TTS synthesis in a separate thread.
    """
    global engine, stream
    try:
        print(f"Thread started for voice '{selected_voice.prompt_wav_path}'. Synthesizing text...")

        # 1. Set the voice on the shared engine instance for this specific request.
        engine.set_voice(selected_voice)

        # 2. Define a callback to push audio chunks into the queue.
        def on_audio_chunk(chunk: bytes):
            audio_queue.put(chunk)

        # 3. Feed the text to the stream and start playing (synthesis).
        #    The `on_audio_chunk` callback will be fired for each chunk.
        stream.feed(text_to_speak)
        stream.play(
            on_audio_chunk=on_audio_chunk,
            comma_silence_duration=0.3,
            sentence_silence_duration=0.6,
            default_silence_duration=0.6,
            muted=True  # We don't want the server to play the audio, just generate it.
        )
        print("Synthesis finished.")

    except Exception as e:
        print(f"An error occurred during TTS processing: {e}")
    finally:
        # 4. Signal the end of the stream by putting None in the queue.
        audio_queue.put(None)
        # 5. Release the semaphore so the next request can be processed.
        tts_semaphore.release()
        print("Thread finished and semaphore released.")


# --- Audio Generator for Streaming Response ---
def audio_chunk_generator(audio_queue: queue.Queue):
    """
    Yields audio chunks from the queue as they become available.
    This function is used by the StreamingResponse.
    """
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break  # End of stream
        yield chunk
    print("Streaming response generator finished.")


# --- FastAPI Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the TTS engine.
    """
    global engine, stream, AVAILABLE_VOICES

    print("--- Server Starting Up ---")

    if not ZIPVOICE_PROJECT_ROOT or not os.path.isdir(ZIPVOICE_PROJECT_ROOT):
        raise RuntimeError(
            "The 'ZIPVOICE_PROJECT_ROOT' environment variable is not set or is not a valid directory."
        )

    # 1. Create the ZipVoiceVoice objects from the configured paths and text.
    print("Loading voice prompts...")
    try:
        voice_alpha = ZipVoiceVoice(
            prompt_wav_path=VOICE_ALPHA_WAV_PATH,
            prompt_text=VOICE_ALPHA_PROMPT_TEXT
        )
        voice_beta = ZipVoiceVoice(
            prompt_wav_path=VOICE_BETA_WAV_PATH,
            prompt_text=VOICE_BETA_PROMPT_TEXT
        )
        AVAILABLE_VOICES = {
            "alpha-warm": voice_alpha,
            "beta-intense": voice_beta,
        }
        print(f"Loaded {len(AVAILABLE_VOICES)} voices: {list(AVAILABLE_VOICES.keys())}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find a voice prompt file: {e}. Make sure paths are correct.")


    # 2. Initialize the ZipVoiceEngine with the first voice as the default.
    print("Initializing ZipVoiceEngine...")
    engine = ZipVoiceEngine(
        zipvoice_root=ZIPVOICE_PROJECT_ROOT,
        voice=voice_alpha,
        model_name="zipvoice",
        device="cuda" if "cuda" in os.getenv("DEVICE", "cuda") else "cpu" # Prefer CUDA
    )

    # 3. Create the TextToAudioStream.
    stream = TextToAudioStream(engine, muted=True)

    # 4. Warm up the engine to reduce latency on the first request.
    print("Warming up the engine...")
    stream.feed("Server is now ready.").play(muted=True)

    print("--- Server Ready ---")

    yield  # The application is now running

    # --- Shutdown Logic ---
    print("--- Server Shutting Down ---")
    if engine:
        engine.shutdown()
        print("ZipVoiceEngine shut down successfully.")
    print("--- Shutdown Complete ---")


# --- FastAPI App and Endpoint ---
app = FastAPI(lifespan=lifespan)

@app.post("/api/c3BlZWNo")
async def create_speech(request: TTSRequest):
    """
    Accepts text and a voice name, and streams back raw PCM audio.
    """
    # 1. Check if the TTS engine is busy. If so, reject the request immediately.
    if not tts_semaphore.acquire(blocking=False):
        print("Request rejected: TTS service is busy.")
        raise HTTPException(
            status_code=503,
            detail="TTS service is busy. Please try again later.",
            headers={"Retry-After": "5"}
        )

    # 2. Validate the requested voice.
    selected_voice = AVAILABLE_VOICES.get(request.voice)
    if not selected_voice:
        tts_semaphore.release()  # Release the lock before raising the error
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{request.voice}' not found. Available voices: {list(AVAILABLE_VOICES.keys())}"
        )

    # 3. Create a queue to pass audio data from the processing thread to this endpoint.
    audio_queue = queue.Queue()

    # 4. Start the TTS processing in a separate thread.
    print(f"Received request for voice '{request.voice}'. Starting processing thread.")
    threading.Thread(
        target=process_tts_request,
        args=(request.text, selected_voice, audio_queue),
        daemon=True
    ).start()

    # 5. Return a streaming response that yields chunks from our generator.
    # ZipVoice outputs 16-bit PCM audio at a 24000 Hz sample rate.
    return StreamingResponse(
        audio_chunk_generator(audio_queue),
        media_type="audio/pcm; rate=24000; bit-depth=16; channels=1"
    )


# --- Main Guard ---
if __name__ == "__main__":
    print("Starting FastAPI server for ZipVoice TTS...")
    if not ZIPVOICE_PROJECT_ROOT:
        print("\nERROR: The 'ZIPVOICE_PROJECT_ROOT' environment variable is not set.")
        print("Please set it to the path of your ZipVoice project clone before running.")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=9086)