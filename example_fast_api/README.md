# Text-to-Speech FastAPI Server

## Overview

This repository contains a FastAPI server that serves as a wrapper for Realtime Text-to-Speech (TTS) functionality. It allows users to synthesize text into speech in real-time. The server handles various routes for managing engines, voices, and synthesizing text.

## Installation

```
pip install fastapi
```

## Components

### `server.py`

This file contains the FastAPI wrapper for Realtime TTS. It is responsible for handling incoming requests, synthesizing text into speech, and sending wave headers to ensure compatibility with browsers. Example usage:

```http
GET http://localhost:8000/tts?text=Hello%20World
```

### `client.py`

This file provides a test client in Python for interacting with the server. It can be used to verify the functionality of the server.

## Routes

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

## User Limitation

Currently, the server can only serve one user at a time when the `/tts` route is called. This limitation is primarily due to the Coqui TTS engine's inability to handle multiple synthesis requests in parallel. Potential solutions include:

- Allowing multiple user requests only when using non-Coqui engines.
- Implementing multiprocessing to create multiple Coqui synthesizers. However, this approach is hardware-intensive, as each synthesizer requires approximately 4GB of RAM. For example, a system with a NVIDIA GeForce RTX 4090 could potentially serve up to 6 users in parallel using this method. However, achieving flawless operation with multiprocessing may require substantial effort and optimization.

While the server can process large amounts of text for synthesis, it currently cannot handle text fed in chunk by chunk, as might be required for bidirectional communication with large language models (LLMs) like GPT. I have some solution approaches in mind that could work for this fastapi server but currently I think this might be a functionality for the coming websocket server only.
