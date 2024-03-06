FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
    git \
    software-properties-common \
    build-essential \
    cmake \
    wget \
    unzip \
    openssh-client \
    libopencv-dev \
    libmagic1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-11

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace