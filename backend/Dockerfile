# HEIC support deployment - v1.1
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including Blender and HEIC support)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gcc \
    g++ \
    libgl1-mesa-dev \
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
    libheif1 \
    libheif-dev \
    blender \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install core dependencies first
RUN pip install --upgrade pip

# Install PyTorch (CPU version for Cloud Run reliability)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install compatible ML ecosystem
RUN pip install --no-cache-dir \
    huggingface-hub==0.20.0 \
    transformers==4.36.0 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1

# Install all other dependencies
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
    numpy>=1.26.0 \
    python-multipart>=0.0.9 \
    jq>=1.6.0 \
    typer>=0.9.0 \
    openai>=1.0.0 \
    bcrypt>=4.0.1 \
    pillow>=10.0.0 \
    pillow-heif>=0.13.0 \
    pyheif \
    fal-client \
    mediapipe==0.10.21 \
    trimesh[easy]==4.0.5 \
    open3d==0.19.0 \
    pybullet==3.2.7 \
    opencv-python==4.8.1.78 \
    scikit-image==0.22.0 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    aiofiles \
    psutil>=5.9.0

# Copy application code
COPY . .

# Detect Blender version and configure Python path
RUN BLENDER_VERSION=$(blender --version | head -1 | grep -oP '\d+\.\d+') && \
    echo "Detected Blender version: $BLENDER_VERSION" && \
    export BLENDER_PYTHON_PATH="/usr/share/blender/$BLENDER_VERSION/python/lib/python*/site-packages" && \
    echo "export PYTHONPATH=\$PYTHONPATH:/usr/share/blender/scripts/modules:$BLENDER_PYTHON_PATH" >> /etc/environment

# Test Blender subprocess availability (no bpy import needed)
RUN blender --version && echo "✅ Blender subprocess available"

# Environment configuration
ENV PORT=8000
ENV DISPLAY=:99
ENV BLENDER_HEADLESS=1
ENV BLENDER_USER_CONFIG=/tmp/blender_config
ENV BLENDER_USER_SCRIPTS=/tmp/blender_scripts

EXPOSE $PORT

# Copy and setup startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
