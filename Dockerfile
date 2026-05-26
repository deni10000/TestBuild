# Можно заменить на любой образ; этот — из внутреннего реестра Яндекса,
# собирается быстрее потому, что pytorch уже в слое.
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    build-essential \
    git \
    curl

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --no-cache-dir pip setuptools wheel

RUN python3.10 -m pip install --no-cache-dir \
    torch==2.12.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

RUN python3.10 -m pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "bitsandbytes" \
    "accelerate" \
    "peft" \
    "trl" \
    "xformers" \
    "triton" \
    "sentencepiece" \
    "protobuf"


WORKDIR /workspace
COPY . .

ENTRYPOINT ["python3.10", "solution.py"]
