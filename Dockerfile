FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    CUDAToolkit_ROOT=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Basic utilities and compiler support for Python packages that build
# small C/C++ extensions during installation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN python -m pip uninstall -y spin \
    && python -m pip install --upgrade pip "setuptools<82" wheel \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m pip check

# BuildKit does not mount the host NVIDIA driver. Confirm the torch stack and
# CUDA compiler without initializing the driver.
RUN python -c \
    "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda)" \
    && nvcc --version

CMD ["/bin/bash"]
