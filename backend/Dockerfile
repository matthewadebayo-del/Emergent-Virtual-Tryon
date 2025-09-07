FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
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

# Copy requirements
COPY requirements.txt .

# Install PyTorch 2.2.0+cu118 (minimum available CUDA version)
RUN pip install --no-cache-dir \
    torch==2.2.0+cu118 \
    torchvision==0.17.0+cu118 \
    torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install ML dependencies in exact order to prevent conflicts
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    huggingface-hub==0.20.3 \
    safetensors==0.4.2 \
    accelerate==0.27.2 \
    transformers==4.38.2 \
    diffusers==0.27.2

# Install remaining dependencies
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
    python-jose>=3.3.0 \
    requests>=2.31.0 \
    pandas>=2.2.0 \
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

COPY . .
EXPOSE 8000

ENV DISPLAY=:99
ENV BLENDER_HEADLESS=1
ENV BLENDER_USER_CONFIG=/tmp/blender_config
ENV BLENDER_USER_SCRIPTS=/tmp/blender_scripts
ENV PYTHONPATH="${PYTHONPATH}:/usr/share/blender/scripts/modules"

RUN echo '#!/bin/bash\n\
echo "Starting virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
export DISPLAY=:99\n\
sleep 2\n\
echo "Testing dependencies..."\n\
python3 -c "\
try:\n\
    from huggingface_hub import cached_download\n\
    print(\"cached_download available\")\n\
    from diffusers import StableDiffusionImg2ImgPipeline\n\
    print(\"diffusers imports successfully\")\n\
    import bpy\n\
    print(\"Blender bpy available\")\n\
    import torch\n\
    print(f\"torch version: {torch.__version__}\")\n\
except ImportError as e:\n\
    print(f\"Import error: {e}\")\n\
"\n\
echo "Starting application..."\n\
uvicorn server:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
