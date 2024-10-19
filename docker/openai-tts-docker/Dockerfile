FROM python:3.10-bookworm

ARG OPENAI_API_KEY=""
ARG TTS_FASTAPI_PORT=8501
ARG WORKERS=4

ENV WORKERS=$WORKERS
ENV TTS_FASTAPI_PORT=$TTS_FASTAPI_PORT
ENV OPENAI_API_KEY=$OPENAI_API_KEY

WORKDIR /code

RUN apt-get update

RUN apt-get install -y portaudio19-dev ffmpeg

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

EXPOSE $TTS_FASTAPI_PORT

CMD ["python", "app/main.py"]
