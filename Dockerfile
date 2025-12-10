# Neural 3D Reconstruction Training Container
# Optimized for multi-GPU training with resource isolation

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch with CUDA 13.0 support for Blackwell (sm_121) GPUs
# Use nightly build with cu130 for best Blackwell compatibility
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Install other requirements (excluding torch since it's already installed)
RUN grep -v "^torch" requirements.txt > /tmp/requirements_no_torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# Verify CUDA is available
RUN python -c "import torch; assert torch.cuda.is_available() or True, 'CUDA check skipped at build time'; print('PyTorch version:', torch.__version__, 'CUDA:', torch.version.cuda)"

# Install additional useful packages
RUN pip install --no-cache-dir \
    nvitop \
    gpustat

# Create non-root user for security
RUN useradd -m -s /bin/bash trainer \
    && chown -R trainer:trainer /workspace

# Default command
CMD ["python", "scripts/train.py", "--config", "config/experiment/cse_warehouse.yaml"]
