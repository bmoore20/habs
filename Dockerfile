FROM nvidia/cuda:11.0-base-ubuntu20.04 as cuda-base

LABEL maintainer.name="Tynan Daly" \
      maintainer.email="tynan.s.daly@gmail.com" \
      release="0.0.1"


ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=$PATH:/root/miniconda3/bin
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    sudo \
    wget \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*



# Install python 3.8


FROM cuda-base as python-base


RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh  -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda install python=3.9


FROM python-base as dl-requirements

COPY requirements.txt ./

RUN pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*



RUN mkdir /app
WORKDIR /app
