# 1) Basis-Image
# Using the same CUDA-enabled base image for GPU support.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the working directory
WORKDIR /opt/app-root/src

# 2) System-Pakete installieren
# Installiere notwendige Systemabhängigkeiten. 'git' wurde für das Klonen von ZipVoice hinzugefügt.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      python3 python3-pip python3-dev \
      build-essential \
      libsndfile1 \
      libportaudio2 \
      portaudio19-dev \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) Lokalen RealtimeTTS-Code kopieren und installieren
# Hier wird davon ausgegangen, dass sich Ihr RealtimeTTS-Code in einem Ordner namens "RealtimeTTS" befindet,
# der sich neben Ihrem Dockerfile befindet.
# COPY ./RealtimeTTS/ /opt/app-root/src/RealtimeTTS
COPY ./RealtimeTTS/ /opt/app-root/src/RealtimeTTS/RealtimeTTS
COPY ./README.md /opt/app-root/src/RealtimeTTS/README.md
COPY ./setup.py /opt/app-root/src/RealtimeTTS/setup.py
COPY ./requirements.txt /opt/app-root/src/RealtimeTTS/requirements.txt

RUN pip install --no-cache-dir -e /opt/app-root/src/RealtimeTTS

# 4) ZipVoice-Repository klonen und Abhängigkeiten installieren (NEU)
# Klont das ZipVoice-Repository und installiert die erforderlichen Pakete aus der requirements.txt.
RUN git clone https://github.com/k2-fsa/ZipVoice.git /opt/app-root/src/ZipVoice
RUN pip install --no-cache-dir -r /opt/app-root/src/ZipVoice/requirements.txt

# 5) Optionale k2-Installation für ZipVoice (NEU & EMPFOHLEN)
# Installiert k2 für Training und effizientere Inferenz, wie in der ZipVoice-Dokumentation empfohlen.
# Die Version muss zur PyTorch- und CUDA-Version passen.
RUN pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html

# 6) DeepSpeed und andere Python-Abhängigkeiten installieren
# DeepSpeed-Build-Variablen werden für eine optimierte Installation gesetzt.
ENV DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_CPU_ADAM=0 \
    DS_BUILD_FUSED_ADAM=0 \
    DS_BUILD_UTILS=0 \
    DS_BUILD_OPS=0

ENV APP_CACHE="/opt/app-root/cache"
RUN mkdir -p ${APP_CACHE} && chmod -R 777 ${APP_CACHE}

ENV HOME="${APP_CACHE}"
ENV XDG_CACHE_HOME="${APP_CACHE}/.cache"
ENV TORCH_HOME="${APP_CACHE}/torch"
ENV HF_HOME="${APP_CACHE}/huggingface"
ENV XDG_DATA_HOME="${APP_CACHE}/.local/share"

RUN mkdir -p \
      ${XDG_CACHE_HOME} \
      ${TORCH_HOME} \
      ${HF_HOME} \
      ${XDG_DATA_HOME} \
    && chmod -R 777 ${APP_CACHE}

RUN pip install --no-cache-dir \
      deepspeed \
      fastapi uvicorn python-multipart python-dotenv httpx

# 7) FastAPI-Anwendungscode kopieren
COPY ./docker/zipvoice/entrypoint.sh /opt/app-root/src/entrypoint.sh
COPY ./docker/zipvoice/serve_zipchunks.py /opt/app-root/src/serve_zipchunks.py
COPY ./docker/zipvoice/reference1.wav /opt/app-root/src/reference1.wav
COPY ./docker/zipvoice/reference2.wav /opt/app-root/src/reference2.wav
RUN chmod +x /opt/app-root/src/entrypoint.sh

# Document that the application listens on port 9086 (for ZipVoice => ZV => Z is ASCII 90 and V is ASCII 86)
EXPOSE 9086

# 8) Start-Befehl
# Führt das Entrypoint-Skript beim Starten des Containers aus.
CMD ["/opt/app-root/src/entrypoint.sh"]