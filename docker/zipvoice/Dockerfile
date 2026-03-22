# CUDA-enabled base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /opt/app-root/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libportaudio2 \
    libsndfile1 \
    portaudio19-dev \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install local RealtimeTTS code
COPY ./RealtimeTTS/ /opt/app-root/src/RealtimeTTS/RealtimeTTS
COPY ./README.md /opt/app-root/src/RealtimeTTS/README.md
COPY ./setup.py /opt/app-root/src/RealtimeTTS/setup.py
COPY ./requirements.txt /opt/app-root/src/RealtimeTTS/requirements.txt

RUN pip install --no-cache-dir -e /opt/app-root/src/RealtimeTTS

# Clone ZipVoice repository and install dependencies
RUN git clone https://github.com/k2-fsa/ZipVoice.git /opt/app-root/src/ZipVoice && \
    pip install --no-cache-dir -r /opt/app-root/src/ZipVoice/requirements.txt

# Install k2 for ZipVoice training and efficient inference
RUN pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html

# Configure cache directories and permissions
ENV APP_CACHE="/opt/app-root/cache" \
    HOME="/opt/app-root/cache" \
    XDG_CACHE_HOME="/opt/app-root/cache/.cache" \
    TORCH_HOME="/opt/app-root/cache/torch" \
    HF_HOME="/opt/app-root/cache/huggingface" \
    XDG_DATA_HOME="/opt/app-root/cache/.local/share"

RUN mkdir -p \
    ${APP_CACHE} \
    ${XDG_CACHE_HOME} \
    ${TORCH_HOME} \
    ${HF_HOME} \
    ${XDG_DATA_HOME} \
    && chmod -R 777 ${APP_CACHE}

# Configure DeepSpeed build options and install Python dependencies
ENV DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_CPU_ADAM=0 \
    DS_BUILD_FUSED_ADAM=0 \
    DS_BUILD_UTILS=0 \
    DS_BUILD_OPS=0

RUN pip install --no-cache-dir \
    deepspeed \
    fastapi \
    httpx \
    python-dotenv \
    python-multipart \
    uvicorn

# Copy application files
COPY ./docker/zipvoice/entrypoint.sh /opt/app-root/src/entrypoint.sh
COPY ./docker/zipvoice/serve_zipchunks.py /opt/app-root/src/serve_zipchunks.py
COPY ./docker/zipvoice/reference1.wav /opt/app-root/src/reference1.wav
COPY ./docker/zipvoice/reference2.wav /opt/app-root/src/reference2.wav

RUN chmod +x /opt/app-root/src/entrypoint.sh

# Expose port for ZipVoice service
EXPOSE 9086

# Start the application
CMD ["/opt/app-root/src/entrypoint.sh"]