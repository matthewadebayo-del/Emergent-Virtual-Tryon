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
    # Graphics libraries for Blender
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
    # Blender and virtual display
    blender \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies in single step
COPY requirements.txt .

# Install PyTorch with CUDA first to establish correct index
RUN pip install --no-cache-dir torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies (this will respect the CUDA PyTorch installation)
RUN pip install --no-cache-dir \
    diffusers==0.21.4 \
    transformers==4.35.2 \
    huggingface-hub==0.17.3 \
    accelerate==0.24.1 \
    safetensors==0.4.0

# Install remaining non-ML dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.1 \
    uvicorn==0.25.0 \
    boto3>=1.34.129 \
    requests-oauthlib>=2.0.0 \
    cryptography>=42.0.8 \
    python-dotenv>=1.0.1 \
    pymongo==4.5.0 \
    pydantic>=2.6.4 \
    email-validator>=2.2.0 \
    pyjwt>=2.10.1 \
    passlib>=1.7.4 \
    tzdata>=2024.2 \
    motor==3.3.1 \
    pytest>=8.0.0 \
    black>=24.1.1 \
    isort>=5.13.2 \
    flake8>=7.0.0 \
    mypy>=1.8.0 \
    python-jose>=3.3.0 \
    requests>=2.31.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    python-multipart>=0.0.9 \
    jq>=1.6.0 \
    typer>=0.9.0 \
    openai>=1.0.0 \
    bcrypt>=4.0.1 \
    pillow>=10.0.0 \
    fal-client \
    mediapipe==0.10.21 \
    trimesh[easy]==4.0.5 \
    open3d==0.19.0 \
    pybullet==3.2.7 \
    opencv-python==4.8.1.78 \
    scikit-image==0.22.0 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    aiofiles

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

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🖥️ Starting virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
export DISPLAY=:99\n\
sleep 2\n\
echo "🧪 Testing dependencies..."\n\
python3 -c "\
try:\n\
    from huggingface_hub import cached_download\n\
    print(\"✅ cached_download available\")\n\
    from diffusers import StableDiffusionImg2ImgPipeline\n\
    print(\"✅ diffusers imports successfully\")\n\
    import bpy\n\
    print(\"✅ Blender bpy available\")\n\
    import torch\n\
    print(f\"✅ torch version: {torch.__version__}\")\n\
    print(f\"✅ CUDA available: {torch.cuda.is_available()}\")\n\
except ImportError as e:\n\
    print(f\"❌ Import error: {e}\")\n\
"\n\
echo "🚀 Starting application..."\n\
uvicorn server:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
