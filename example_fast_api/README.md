# Text-to-Speech FastAPI Server

## Overview

This repository contains a FastAPI server that serves as a wrapper for Realtime Text-to-Speech (TTS) functionality. It allows users to synthesize text into speech in real-time. The server handles various routes for managing engines, voices, and synthesizing text.

## Installation

```bash
pip install fastapi uvicorn websockets pyaudio
```

## Components

### `async_server.py`

This file contains the FastAPI server with both HTTP and WebSocket endpoints for Realtime TTS. It is responsible for handling incoming requests, synthesizing text into speech, and streaming audio. Features:

- **HTTP endpoint** (`/tts`): Traditional REST API for text-to-speech
- **WebSocket endpoint** (`/ws`): Real-time bidirectional TTS communication with support for multiple concurrent users

Example HTTP usage:
```http
GET http://localhost:8000/tts?text=Hello%20World
```

Example WebSocket usage:
```python
# Connect to ws://localhost:8000/ws
# Send text messages, receive base64-encoded audio chunks
```

### `server.py`

Legacy synchronous server (use `async_server.py` for production).

### `client.py`

This file provides a HTTP test client in Python for interacting with the `/tts` endpoint.

### `websocket_client.py`

WebSocket demo client that:
- Connects to the `/ws` endpoint
- Sends text messages for TTS synthesis
- Receives and plays audio in real-time
- Optionally saves audio to WAV files

Usage:
```bash
# Start the server first
python async_server.py

# Run the WebSocket client with default test messages
python websocket_client.py

# Or provide custom messages
python websocket_client.py "Hello world" "This is a test"
```

## Routes

### HTTP Endpoints

- `/`: Serves HTML and JavaScript to operate the server.
- `/tts`: Returns audio chunks for text synthesis.
  - Parameters:
    - `text`: Text to synthesize.
- `/engines`: Lists available TTS engines.
  - Parameters: None.
- `/set_engine`: Switches the current TTS engine.
  - Parameters:
    - `engine_name`: Name of the engine to switch to.
- `/voices`: Lists voices available for the current engine.
  - Parameters: None.
- `/setvoice`: Sets the voice for the current engine.
  - Parameters:
    - `voice_name`: Name of the voice to switch to.

### WebSocket Endpoint

- `/ws`: Real-time TTS synthesis via WebSocket
  - **Send**: Plain text messages to synthesize
  - **Receive**: JSON messages with audio chunks:
    ```json
    {
      "audioOutput": {
        "audio": "base64_encoded_audio_data",
        "format": "wav",
        "sampleRate": 24000,
        "isHeader": true  // Only present in first message
      }
    }
    ```
  - **Completion signal**:
    ```json
    {
      "finalOutput": {
        "isFinal": true
      }
    }
    ```
  - **Stop**: Send "stop" message to close connection

## Concurrent User Support

### HTTP Endpoint (`/tts`)

Currently, the HTTP endpoint can only serve one user at a time. This limitation is primarily due to the Coqui TTS engine's inability to handle multiple synthesis requests in parallel. Potential solutions include:

- Allowing multiple user requests only when using non-Coqui engines.
- Implementing multiprocessing to create multiple Coqui synthesizers. However, this approach is hardware-intensive, as each synthesizer requires approximately 4GB of RAM. For example, a system with a NVIDIA GeForce RTX 4090 could potentially serve up to 6 users in parallel using this method. However, achieving flawless operation with multiprocessing may require substantial effort and optimization.

### WebSocket Endpoint (`/ws`)

**The WebSocket endpoint supports multiple concurrent users!** Each WebSocket connection:
- Has its own dedicated request queue
- Processes requests sequentially per connection
- Runs independently from other connections
- Creates dedicated TTS handlers for each synthesis request
