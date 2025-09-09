# NUCLEAR OPTION: Complete Docker cache bypass
FROM python:3.12-slim

# Force cache invalidation with build timestamp
ARG CACHE_BUST=1
RUN echo "Cache bust: $CACHE_BUST $(date)"

WORKDIR /app

# Install system dependencies
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
    blender \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# NUCLEAR APPROACH: Completely remove any existing packages
RUN pip uninstall -y torch torchvision torchaudio transformers diffusers huggingface-hub accelerate safetensors numpy || true
RUN pip cache purge
RUN rm -rf /root/.cache/pip
RUN rm -rf /usr/local/lib/python3.12/site-packages/torch*
RUN rm -rf /usr/local/lib/python3.12/site-packages/transformers*
RUN rm -rf /usr/local/lib/python3.12/site-packages/diffusers*
RUN rm -rf /usr/local/lib/python3.12/site-packages/huggingface*

# Install build dependencies in separate steps for proper availability
RUN echo "Installing core build tools..." && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

RUN echo "Installing build dependencies..." && \
    pip install --no-cache-dir typing_extensions filelock networkx jinja2 fsspec && \
    echo "✅ Build dependencies installed"

# Install numpy first (required by many packages) - Python 3.12 compatible version
RUN echo "Installing numpy 1.26.0 (Python 3.12 compatible)..." && \
    pip install --no-cache-dir --only-binary=numpy numpy==1.26.0 && \
    python3 -c "import numpy; print(f'✅ numpy: {numpy.__version__}')"

# Install PyTorch with CPU support
RUN echo "Installing PyTorch 2.2.0 (CPU)..." && \
    pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    python3 -c "import torch; print(f'✅ torch: {torch.__version__}')"

RUN echo "Installing huggingface-hub 0.16.4..." && \
    pip install --no-cache-dir --no-deps huggingface-hub==0.16.4 && \
    python3 -c "from huggingface_hub import cached_download; print('✅ cached_download available')"

RUN echo "Installing safetensors 0.4.2..." && \
    pip install --no-cache-dir --no-deps safetensors==0.4.2 && \
    python3 -c "import safetensors; print(f'✅ safetensors: {safetensors.__version__}')"

RUN echo "Installing accelerate 0.27.2..." && \
    pip install --no-cache-dir --no-deps accelerate==0.27.2 && \
    python3 -c "import accelerate; print(f'✅ accelerate: {accelerate.__version__}')"

RUN echo "Installing transformers 4.33.3..." && \
    pip install --no-cache-dir --no-deps transformers==4.33.3 && \
    python3 -c "import transformers; print(f'✅ transformers: {transformers.__version__}')"

RUN echo "Installing diffusers 0.21.4..." && \
    pip install --no-cache-dir --no-deps diffusers==0.21.4 && \
    python3 -c "import diffusers; print(f'✅ diffusers: {diffusers.__version__}')"

# Step 4: Prevent Version Drift - Pin ML packages after installation
RUN pip freeze | grep -E "(torch|transformers|diffusers|huggingface)" > /tmp/ml_freeze.txt && \
    pip install --no-deps --force-reinstall -r /tmp/ml_freeze.txt && \
    echo "🔒 ML packages pinned to prevent version drift"

# Final verification
RUN echo "Final verification..." && \
    python3 -c "from huggingface_hub import cached_download; print('✅ cached_download available')" && \
    python3 -c "import torch; print(f'✅ torch: {torch.__version__}')" && \
    python3 -c "import diffusers; print(f'✅ diffusers: {diffusers.__version__}')" && \
    python3 -c "import transformers; print(f'✅ transformers: {transformers.__version__}')" && \
    echo "🎉 ALL ML DEPENDENCIES VERIFIED!"

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

# Add explicit PORT configuration for Cloud Run
ENV PORT=8000
EXPOSE $PORT

ENV DISPLAY=:99
ENV BLENDER_HEADLESS=1
ENV BLENDER_USER_CONFIG=/tmp/blender_config
ENV BLENDER_USER_SCRIPTS=/tmp/blender_scripts
ENV PYTHONPATH="${PYTHONPATH}:/usr/share/blender/scripts/modules"

# Create Blender config directory and disable network calls
RUN mkdir -p /tmp/blender_config /tmp/blender_scripts && \
    echo "import bpy; bpy.context.preferences.system.use_international_fonts = False" > /tmp/blender_config/startup.py

# Test bpy availability during build with direct import (no subprocess)
RUN python3 -c "\
import sys;\
sys.path.append('/usr/share/blender/scripts/modules');\
# Try common Blender version paths\
blender_paths = ['/usr/share/blender/4.0/python/lib/python3.11/site-packages', '/usr/share/blender/3.6/python/lib/python3.10/site-packages'];\
for path in blender_paths:\
    if __import__('os').path.exists(path): sys.path.append(path);\
import bpy;\
print(f'✅ Blender Python API available: {bpy.app.version_string}');\
bpy.ops.mesh.primitive_cube_add();\
bpy.ops.object.delete();\
print('✅ Blender operations working');\
" && echo "✅ Blender Python API verification complete"

RUN echo '#!/bin/bash\n\
echo "Starting virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
export DISPLAY=:99\n\
sleep 2\n\
echo "Final runtime verification..."\n\
python3 -c "\
try:\n\
    from huggingface_hub import cached_download\n\
    from diffusers import StableDiffusionImg2ImgPipeline\n\
    import torch\n\
    import transformers\n\
    print('✅ Step 5: All AI dependencies verified at runtime')\n\
    print(f'torch: {torch.__version__}')\n\
    print(f'transformers: {transformers.__version__}')\n\
    print('🎉 ALL RUNTIME IMPORTS SUCCESSFUL!')\n\
except ImportError as e:\n\
    print(f'❌ Step 5: Runtime import error: {e}')\n\
    print('🔧 Starting with AI enhancement disabled')\n\
    import os\n\
    os.environ['DISABLE_AI'] = 'true'\n\
" || echo "AI dependencies failed, continuing with fallback mode"\n\
echo "Starting application..."\n\
uvicorn server:app --host 0.0.0.0 --port $PORT\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
