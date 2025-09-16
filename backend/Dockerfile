FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including Blender)
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

# Create startup script with proper error handling
RUN echo '#!/bin/bash\n\
echo "Starting virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
export DISPLAY=:99\n\
sleep 2\n\
\n\
# Store original PYTHONPATH\n\
export PYTHONPATH_ORIGINAL="$PYTHONPATH"\n\
\n\
echo "Configuring Blender Python paths..."\n\
BLENDER_VERSION=$(blender --version | head -1 | grep -oP "\\d+\\.\\d+" || echo "4.0")\n\
echo "Detected Blender version: $BLENDER_VERSION"\n\
\n\
# Initialize PYTHONPATH with Blender scripts\n\
export PYTHONPATH="/usr/share/blender/scripts/modules"\n\
\n\
# Add version-specific paths\n\
for path in /usr/share/blender/$BLENDER_VERSION/python/lib/python*/site-packages; do\n\
    if [ -d "$path" ]; then\n\
        export PYTHONPATH="$PYTHONPATH:$path"\n\
        echo "Added version-specific path: $path"\n\
    fi\n\
done\n\
\n\
# Add fallback paths for any Blender version\n\
for path in /usr/share/blender/*/python/lib/python*/site-packages; do\n\
    if [ -d "$path" ] && [[ ":$PYTHONPATH:" != *":$path:"* ]]; then\n\
        export PYTHONPATH="$PYTHONPATH:$path"\n\
        echo "Added fallback path: $path"\n\
    fi\n\
done\n\
\n\
# Add system Python path as fallback\n\
export PYTHONPATH="$PYTHONPATH:$PYTHONPATH_ORIGINAL"\n\
echo "Final PYTHONPATH: $PYTHONPATH"\n\
\n\
echo "Testing dependencies..."\n\
python3 -c "\
import sys\n\
\n\
# Test core 3D dependencies (critical)\n\
try:\n\
    import trimesh, mediapipe, open3d\n\
    print('Core 3D dependencies: OK')\n\
except ImportError as e:\n\
    print(f'Critical 3D dependency missing: {e}')\n\
    sys.exit(1)\n\
\n\
# Test Blender subprocess (no bpy import needed)\n\
import subprocess\n\
try:\n\
    result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)\n\
    if result.returncode == 0:\n\
        version = result.stdout.split('\\n')[0] if result.stdout else 'Unknown'\n\
        print(f'Blender subprocess: OK - {version}')\n\
    else:\n\
        print('Blender subprocess: Failed - using fallback rendering')\n\
except Exception as e:\n\
    print(f'Blender subprocess unavailable: {e} - using fallback rendering')\n\
\n\
# Test AI dependencies (non-critical)\n\
try:\n\
    import torch, transformers, diffusers\n\
    print('AI dependencies: OK')\n\
except ImportError as e:\n\
    print(f'AI dependencies unavailable: {e}')\n\
    print('AI enhancement will be disabled')\n\
    import os\n\
    os.environ['DISABLE_AI_ENHANCEMENT'] = 'true'\n\
\n\
print('Startup checks complete')\n\
"\n\
\n\
echo "Starting application..."\n\
uvicorn server:app --host 0.0.0.0 --port $PORT\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
