# Cloud Run PyTorch/Diffusers Deployment Guide

## Common Issues with PyTorch on Google Cloud Run

### 1. Container Size Limits
- Cloud Run has a 32GB container size limit
- PyTorch with CUDA can easily exceed this
- Solution: Use CPU-only PyTorch builds

### 2. Memory Constraints
- Default Cloud Run memory: 512MB-8GB
- PyTorch models can require 2-4GB+ just to load
- Solution: Increase memory allocation and use lazy loading

### 3. Cold Start Timeouts
- Cloud Run has 240s startup timeout
- Loading large ML models can exceed this
- Solution: Implement lazy loading and model caching

## Optimized Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app
WORKDIR /app

# Set environment variables for memory management
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch_cache
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# Create cache directories
RUN mkdir -p /tmp/torch_cache /tmp/hf_cache /tmp/transformers_cache

EXPOSE 8080
CMD ["python", "server.py"]
```

## Environment Variables for Memory Management

```yaml
# cloudbuild.yaml
env:
  - name: PYTHONUNBUFFERED
    value: "1"
  - name: TORCH_HOME
    value: "/tmp/torch_cache"
  - name: HF_HOME
    value: "/tmp/hf_cache"
  - name: TRANSFORMERS_CACHE
    value: "/tmp/transformers_cache"
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: "max_split_size_mb:128"
  - name: OMP_NUM_THREADS
    value: "2"
```

## Lazy Loading Strategy

```python
# model_manager.py
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self._diffusion_pipeline = None
        self._torch_available = None
        
    def _check_torch_availability(self) -> bool:
        if self._torch_available is None:
            try:
                import torch
                import diffusers
                self._torch_available = True
                logger.info("✅ PyTorch and diffusers available")
            except ImportError as e:
                logger.warning(f"⚠️ PyTorch/diffusers not available: {e}")
                self._torch_available = False
        return self._torch_available
    
    def get_diffusion_pipeline(self):
        if not self._check_torch_availability():
            return None
            
        if self._diffusion_pipeline is None:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                import torch
                
                logger.info("Loading Stable Diffusion pipeline...")
                self._diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1",
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                self._diffusion_pipeline = self._diffusion_pipeline.to("cpu")
                logger.info("✅ Stable Diffusion pipeline loaded")
                
            except Exception as e:
                logger.error(f"❌ Failed to load diffusion pipeline: {e}")
                self._diffusion_pipeline = None
                
        return self._diffusion_pipeline
```

## Requirements.txt for CPU-only PyTorch

```txt
# Core FastAPI dependencies
fastapi==0.110.1
uvicorn[standard]==0.25.0
python-dotenv==1.0.1

# AI/ML dependencies (CPU-only)
torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
diffusers==0.30.0
transformers>=4.30.0
accelerate>=0.20.0

# Image processing
pillow>=10.0.0
opencv-python-headless==4.8.1.78

# 3D processing
trimesh[easy]==4.0.5
numpy>=1.26.0
scipy==1.11.4

# Database
pymongo==4.5.0
motor==3.3.1
```

## Common Deployment Issues & Solutions

### Issue 1: "No module named 'torch'"
**Cause:** PyTorch not installed or wrong index URL
**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: Container size too large
**Cause:** CUDA dependencies included
**Solution:** Use CPU-only builds and multi-stage Docker

### Issue 3: Memory allocation errors
**Cause:** Insufficient memory allocation
**Solution:**
```yaml
resources:
  limits:
    memory: 4Gi
    cpu: 2000m
```

### Issue 4: Cold start timeouts
**Cause:** Loading models during startup
**Solution:** Implement lazy loading and health check delays

### Issue 5: "RuntimeError: No CUDA devices available"
**Cause:** Code trying to use CUDA on CPU-only environment
**Solution:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Cloud Run Configuration

```yaml
# cloudbuild-backend.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/virtualfit-backend', '.']
    dir: 'backend'
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/virtualfit-backend']
  
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'virtualfit-backend'
      - '--image=gcr.io/$PROJECT_ID/virtualfit-backend'
      - '--platform=managed'
      - '--region=us-central1'
      - '--allow-unauthenticated'
      - '--memory=4Gi'
      - '--cpu=2'
      - '--timeout=900'
      - '--max-instances=10'
      - '--set-env-vars=PYTHONUNBUFFERED=1,TORCH_HOME=/tmp/torch_cache'
```

## Alternative Deployment Options

### Option 1: Vertex AI Endpoints
- Better for ML workloads
- Higher memory/CPU limits
- Built-in model serving

### Option 2: Google Compute Engine
- Full control over environment
- No container size limits
- Can use GPU instances

### Option 3: Google Kubernetes Engine
- Better resource management
- Horizontal scaling
- Persistent storage for models

## Monitoring & Debugging

```python
# Add to server.py
import psutil
import logging

@app.get("/debug/system")
async def debug_system():
    return {
        "memory_usage": psutil.virtual_memory()._asdict(),
        "cpu_usage": psutil.cpu_percent(),
        "torch_available": torch.cuda.is_available() if 'torch' in globals() else False,
        "torch_version": torch.__version__ if 'torch' in globals() else None,
    }
```

## Best Practices

1. **Use CPU-only PyTorch** for Cloud Run
2. **Implement lazy loading** for all ML models
3. **Set appropriate memory limits** (4-8GB for ML workloads)
4. **Use multi-stage Docker builds** to reduce image size
5. **Cache models in /tmp** to avoid re-downloading
6. **Monitor memory usage** and adjust limits accordingly
7. **Implement health checks** with appropriate delays
8. **Use environment variables** for configuration
