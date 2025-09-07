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
    libgl1-mesa-dri \
    libglu1-mesa-dev \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxrender1 \
    libgomp1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libx11-6 \
    libxss1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libasound2 \
    libxtst6 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libfreetype6 \
    libfontconfig1 \
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
    xformers==0.0.22.post7

# Install remaining dependencies (excluding ML packages already installed)
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables for headless Blender
ENV DISPLAY=:99
ENV BLENDER_HEADLESS=1
ENV BLENDER_USER_CONFIG=/tmp/blender_config
ENV BLENDER_USER_SCRIPTS=/tmp/blender_scripts
ENV PYTHONPATH="${PYTHONPATH}:/usr/share/blender/scripts/modules"

# Create startup script for proper virtual display initialization
RUN echo '#!/bin/bash\n\
echo "🖥️ Starting virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
export DISPLAY=:99\n\
sleep 2\n\
echo "🧪 Testing Blender availability..."\n\
python3 -c "\
import sys\n\
try:\n\
    import bpy\n\
    print(\"✅ Blender Python API (bpy) is available\")\n\
    print(f\"Blender version: {bpy.app.version_string}\")\n\
except ImportError as e:\n\
    print(f\"❌ Blender Python API not available: {e}\")\n\
"\n\
echo "🚀 Starting application..."\n\
uvicorn server:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

# Run the application with proper Blender initialization
CMD ["/start.sh"]
