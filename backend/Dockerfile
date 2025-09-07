# Dockerfile for VirtualFit Backend (FastAPI)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyBullet, 3D libraries, and Blender
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gcc \
    g++ \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libx11-6 \
    libxss1 \
    libxcomposite1 \
    libasound2 \
    libxtst6 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    blender \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies with CUDA support
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Then install other ML dependencies in one step to avoid conflicts
RUN pip install \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors==0.4.1 \
    xformers==0.0.22.post7 \
    bpy==4.0.0

# Install remaining dependencies (excluding ML packages already installed)
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables for headless Blender
ENV DISPLAY=:99
ENV BLENDER_USER_CONFIG=/tmp/blender_config
ENV BLENDER_USER_SCRIPTS=/tmp/blender_scripts

# Run the application with virtual display for Blender
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 & uvicorn server:app --host 0.0.0.0 --port 8000"]
