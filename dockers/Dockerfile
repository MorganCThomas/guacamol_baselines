FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        wget \
        curl \
        bash \
        bzip2 \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MiniConda
RUN curl -LO --silent https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}

# python deps
COPY requirements.txt /tmp
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Add the source code
RUN mkdir -p /app
ADD . /app

# Launch inside the folder
WORKDIR /app/

# install frag_gt
RUN pip install --upgrade pip && pip install --no-cache-dir /app/frag_gt/