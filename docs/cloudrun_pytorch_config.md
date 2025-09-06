# Cloud Run PyTorch/Diffusers Deployment Guide

## 🚨 Common Cloud Run + PyTorch Issues

1. **Container size limits** (10GB max)
2. **Memory limits** (32GB max for GPU instances)
3. **Cold start timeouts** with large models
4. **CUDA version mismatches**
5. **Model download timeouts**
6. **Build timeouts** during pip install

## 🔧 Optimized Dockerfile for Cloud Run

### Multi-Stage Build (Recommended)

```dockerfile
# Stage 1: Build environment with all build tools
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with specific CUDA version (crucial for Cloud Run GPU)
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other ML dependencies
RUN pip install \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    xformers==0.0.22.post7 \
    --no-deps

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download models during build (crucial for avoiding runtime timeouts)
RUN python3.11 -c "
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import os

# Set cache directory
cache_dir = '/opt/model_cache'
os.makedirs(cache_dir, exist_ok=True)

print('Downloading Stable Diffusion models...')

# Download models to cache
try:
    pipe1 = StableDiffusionImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1',
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False
    )
    print('✅ SD img2img model cached')
    
    pipe2 = StableDiffusionInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-inpainting', 
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False
    )
    print('✅ SD inpainting model cached')
    
    # Clear GPU memory
    del pipe1, pipe2
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f'Model download error: {e}')
    # Don't fail build if models can't be downloaded
"

# Stage 2: Runtime environment (much smaller)
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/opt/venv/bin:$PATH"
ENV TRANSFORMERS_CACHE=/opt/model_cache
ENV HF_HOME=/opt/model_cache
ENV TORCH_HOME=/opt/model_cache

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment and models from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model_cache /opt/model_cache

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /opt/model_cache
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python3.11 -c "import torch; print('✅ PyTorch OK')" || exit 1

# Command
CMD ["python3.11", "-m", "src.api.main"]
```

## ⚙️ Cloud Run Configuration Settings

### 1. Cloud Run Service Configuration

```yaml
# cloudrun-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: virtual-tryon-api
  annotations:
    # Critical: Set higher timeout
    run.googleapis.com/timeout: "3600s"
    # Disable CPU throttling for ML workloads
    run.googleapis.com/cpu-throttling: "false"
    # Enable GPU execution environment
    run.googleapis.com/execution-environment: gpu
    
spec:
  template:
    metadata:
      annotations:
        # Resource allocation (critical for PyTorch)
        run.googleapis.com/memory: "32Gi"        # Max memory for GPU instances
        run.googleapis.com/cpu: "8"              # Max CPU for GPU instances
        run.googleapis.com/gpu-type: "nvidia-l4" # or nvidia-t4
        run.googleapis.com/gpu-count: "1"
        
        # Startup configuration
        run.googleapis.com/startup-cpu-boost: "true"
        run.googleapis.com/max-scale: "10"
        run.googleapis.com/min-scale: "0"        # Scale to zero when idle
        
        # Container configuration
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "0"
        
    spec:
      containerConcurrency: 1                    # One request per container (GPU limitation)
      timeoutSeconds: 3600                       # 1 hour request timeout
      
      containers:
      - image: gcr.io/PROJECT_ID/virtual-tryon:latest
        ports:
        - containerPort: 8000
        
        resources:
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
          requests:
            memory: "16Gi"                       # Minimum for PyTorch/diffusers
            cpu: "4"
            
        env:
        # CUDA configuration
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
          
        # PyTorch configuration
        - name: TORCH_HOME
          value: "/opt/model_cache"
        - name: TRANSFORMERS_CACHE
          value: "/opt/model_cache"
        - name: HF_HOME
          value: "/opt/model_cache"
          
        # Memory optimization
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:128"
        - name: OMP_NUM_THREADS
          value: "4"
        
        # Startup probe (critical for large models)
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 30
          failureThreshold: 10                   # Allow 5 minutes for startup
          
        # Liveness probe
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 60
          timeoutSeconds: 30
          
        # Readiness probe
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          periodSeconds: 30
          timeoutSeconds: 30
```

## 🎯 Model Loading Optimization

### Lazy Loading with Caching

```python
# src/core/model_manager.py
import os
import torch
from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class OptimizedModelManager:
    """Optimized model manager for Cloud Run deployment"""
    
    def __init__(self):
        self.cache_dir = os.getenv('TORCH_HOME', '/opt/model_cache')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._models = {}
        
        # Configure PyTorch for Cloud Run
        if torch.cuda.is_available():
            # Optimize CUDA memory allocation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        logger.info(f"Model manager initialized - Device: {self.device}")
    
    @lru_cache(maxsize=None)
    def get_stable_diffusion_img2img(self):
        """Load SD img2img model with optimization"""
        if 'sd_img2img' not in self._models:
            logger.info("Loading Stable Diffusion img2img model...")
            
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                
                # Load with optimizations
                self._models['sd_img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1",
                    torch_dtype=torch.float16,          # Half precision
                    cache_dir=self.cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,               # Faster loading
                    variant="fp16"                      # Use fp16 variant if available
                ).to(self.device)
                
                # Enable memory efficient attention
                self._models['sd_img2img'].enable_attention_slicing()
                self._models['sd_img2img'].enable_vae_slicing()
                
                if hasattr(self._models['sd_img2img'], 'enable_xformers_memory_efficient_attention'):
                    try:
                        self._models['sd_img2img'].enable_xformers_memory_efficient_attention()
                        logger.info("✅ xFormers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable xFormers: {e}")
                
                logger.info("✅ SD img2img model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load SD model: {e}")
                raise
        
        return self._models['sd_img2img']
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        for model_name, model in self._models.items():
            if hasattr(model, 'to'):
                model.to('cpu')
            del model
            logger.info(f"Cleared model: {model_name}")
        
        self._models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✅ Model cache cleared")
```

## 🚨 Common Deployment Issues & Solutions

### Issue 1: "Container failed to start" with PyTorch

**Cause**: CUDA version mismatch or missing drivers
**Solution**:
```dockerfile
# Use exact CUDA version that matches Cloud Run GPU
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install exact PyTorch version
RUN pip install torch==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "Model loading timeout"

**Cause**: Models downloading during runtime
**Solution**: Pre-download models during build
```dockerfile
# In Dockerfile - download models during build
RUN python -c "
from diffusers import StableDiffusionImg2ImgPipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1',
    cache_dir='/opt/model_cache'
)
"
```

### Issue 3: "Out of memory" errors

**Cause**: Insufficient memory allocation
**Solutions**:
```yaml
# In Cloud Run config
run.googleapis.com/memory: "32Gi"  # Use maximum

# In code - optimize memory usage
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
torch.cuda.set_per_process_memory_fraction(0.9)
```

This guide provides comprehensive solutions for deploying PyTorch/Diffusers applications on Cloud Run with proper GPU support and optimization.
