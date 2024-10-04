## Limitations
Only Supports OpenAI currently

## How to Build & Run
1. ```git clone https://github.com/KoljaB/RealtimeTTS.git```
2. ```cd RealtimeTTS/docker/openai-tts-docker```
3. Edit ```OPENAI_API_KEY``` in ```Dockerfile``` under ```RealtimeTTS/docker/openai-tts-docker``` for TTS engine
3. ```sudo docker build -t realtime .```
4. ```sudo docker run --privileged -p 8501:8501 realtime```

RealtimeTTS requires [Docker](https://www.docker.com/) Docker version 24+ to run.

## Docs
```/tts``` endpoint sends back audio chunks to be played on the client side. An Example is shared here - [client.py](https://github.com/KoljaB/RealtimeTTS/blob/master/example_fast_api/client.py)
