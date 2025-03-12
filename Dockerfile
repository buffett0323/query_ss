# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install essential dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    lsof \
    fuse \
    libnccl-dev \
    libibverbs1 \
    sshfs \
    && rm -rf /var/lib/apt/lists/*


# Set Python 3.12 as the default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3


# ✅ Manually install distutils using ensurepip and pip
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools


# Install PyTorch and dependencies
RUN pip install torch torchvision torchaudio


# Set the working directory inside the container
WORKDIR /app
COPY /simsiam /app/simsiam
COPY requirements.txt /app

RUN pip install -r requirements.txt
RUN pip3 install --upgrade six

# Install dependencies including pycairo
RUN apt-get update && apt-get install -y \
    python3-gi \
    gir1.2-gtk-3.0 \
    libcairo2-dev \
    python3-cairo \
    && rm -rf /var/lib/apt/lists/*


RUN wandb login $WANDB_API_KEY

RUN mkdir -p /mnt/nas
CMD ["/bin/bash"]