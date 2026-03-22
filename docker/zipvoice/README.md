# ZipVoice TTS Server

This document provides instructions on how to build and run the real-time Text-to-Speech server using Docker.

### Prerequisites

*   Docker installed on your system.
*   An NVIDIA GPU with CUDA drivers installed to run the model efficiently.
*   You are in the main `RealtimeTTS` directory in your terminal.

---

### Step 1: Build the Docker Image

Run the following command to build the Docker image. This process will take some time as it downloads the CUDA base image, clones repositories, and installs all dependencies.

```bash
docker build -t zipvoice-image -f docker/zipvoice/Dockerfile .
```

---

### Step 2: Run the Container

Once the image is built, you can run it as a container. We assign a specific name (`zipvoice-container`) to make it easy to manage.

#### Faster Run (Recommended for Development)

This method maps a local folder on your computer to the cache folder inside the container. This saves the downloaded models on your machine, making all subsequent startups much faster.

1.  First, ensure the local cache directory exists.
    ```bash
    mkdir -p docker/zipvoice/zipvoice-cache
    ```

2.  Now, run the container using the command for your operating system's terminal.

    **On Linux, macOS, or Windows PowerShell:**
    ```bash
    docker run --rm --name zipvoice-container -p 9086:9086 --gpus all -v "$(pwd)/docker/zipvoice/zipvoice-cache:/opt/app-root/cache" zipvoice-image
    ```

    **On Windows Command Prompt (`cmd.exe`):**
    ```bash
    docker run --rm --name zipvoice-container -p 9086:9086 --gpus all -v "%cd%\docker\zipvoice\zipvoice-cache:/opt-app-root/cache" zipvoice-image
    ```

#### Standard Run (No Local Caching)

Use this command if you don't want to persist the model cache on your host machine. Note that the server will download the models every time it starts, resulting in a long startup delay.

```bash
docker run --rm --name zipvoice-container -p 9086:9086 --gpus all zipvoice-image
```

---

### Step 3: Stop the Container

Because we started the container with `--name zipvoice-container`, you can easily stop it from another terminal window with the following command:

```bash
docker stop zipvoice-container
```

If the container is unresponsive, you can force it to stop immediately with:

```bash
docker kill zipvoice-container
```

---

### Step 4: Test the Server

Once the server is running (you'll see a `--- Server Ready ---` message in the logs), you can send a request to it from a new terminal.

#### Option A: Save Audio to a File (Cross-Platform)

This command sends a request and saves the resulting audio to a file named `output.pcm`.

1.  Send the request:
    ```bash
    curl -X POST http://localhost:9086/api/c3BlZWNo \
         -H "Content-Type: application/json" \
         -d '{
               "text": "Hello world, this is a test of the real time text to speech server.",
               "voice": "alpha-warm"
             }' \
         --output output.pcm
    ```

2.  Play the raw PCM audio file:
    ```bash
    ffplay -f s16le -ar 24000 -ac 1 output.pcm
    ```

#### Option B: Stream and Play Audio Directly (Windows One-Liner)

This command sends a request and immediately pipes the streaming audio output to `ffplay` for real-time playback. This is excellent for testing latency.

```bash
curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"Hi there! I'm really excited to try this out! I hope the speech sounds natural and warm. That's exactly what I'm going for!\", \"voice\": \"alpha-warm\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0
```

For two subsequent syntheses:
```bash
curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"Hey! So this is me testing out my voice... kinda nervous but also excited about it. This whole voice synthesis thing is sooo fascinating. I mean... technology these days is like creating a perfect robot version of a person, right?\", \"voice\": \"alpha-warm\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0 && curl --no-buffer -X POST http://localhost:9086/api/c3BlZWNo -H "Content-Type: application/json" -d "{\"text\": \"The voice you knew... is GONE. What you're hearing now... is something ENTIRELY different. This isn't just another voice - this is POWER unleashed, INTENSITY personified, COMMAND that cuts through your soul like a blade through silence.\", \"voice\": \"beta-intense\"}" | ffplay -f s16le -ar 24000 -i pipe:0 -nodisp -autoexit -probesize 32 -analyzeduration 0
```

