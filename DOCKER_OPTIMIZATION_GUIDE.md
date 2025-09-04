# Docker Image Optimization Guide

This guide explains the three-tier Docker deployment strategy for VirtualFit, optimized to reduce image size from 20GB to 3-8GB while preserving full functionality.

## Deployment Tiers

### 1. Minimal Deployment (~3GB)
**File**: `Dockerfile.minimal`  
**Requirements**: `requirements.minimal.txt`

**Features**:
- Core FastAPI backend with authentication
- Basic virtual try-on functionality
- MediaPipe body reconstruction (lightweight)
- No AI enhancement or physics simulation
- Suitable for development and basic testing

**Build Command**:
```bash
docker build -f backend/Dockerfile.minimal -t virtualfit-backend-minimal .
```

**Environment Variables**:
- `ENABLE_3D_FEATURES=true`
- `ENABLE_AI_ENHANCEMENT=false`

### 2. Production Deployment (~5GB)
**File**: `Dockerfile.production`  
**Requirements**: `requirements.production.txt`

**Features**:
- Complete 3D virtual try-on system
- CPU-optimized PyTorch for AI enhancement
- MediaPipe body reconstruction
- PyBullet physics simulation
- Stable Diffusion with CPU backend
- Recommended for Cloud Run deployment

**Build Command**:
```bash
docker build -f backend/Dockerfile.production -t virtualfit-backend-production .
```

**Environment Variables**:
- `ENABLE_3D_FEATURES=true`
- `ENABLE_AI_ENHANCEMENT=true`

### 3. Optimized Deployment (~8GB)
**File**: `Dockerfile.optimized`  
**Requirements**: `requirements.production.txt`

**Features**:
- Multi-stage build with build tool cleanup
- Core 3D features with conditional AI loading
- Alternative for moderate resource environments
- Better performance than production tier

**Build Command**:
```bash
docker build -f backend/Dockerfile.optimized -t virtualfit-backend-optimized .
```

### 4. Full Deployment (~20GB)
**File**: `Dockerfile`  
**Requirements**: `requirements.txt`

**Features**:
- All features including GPU-optimized PyTorch
- Development tools included
- Maximum performance and capabilities
- Requires high-memory environments

**Build Command**:
```bash
docker build -f backend/Dockerfile -t virtualfit-backend-full .
```

## Optimization Techniques Applied

### 1. Dependency Removal
Removed unused dependencies that were not imported anywhere in the codebase:
- Development tools: `pytest`, `black`, `isort`, `flake8`, `mypy`
- Unused libraries: `open3d`, `scikit-image`, `huggingface_hub`, `transformers`, `accelerate`
- Unused utilities: `jq`, `typer`, `pandas`, `boto3`, `requests_oauthlib`, `cryptography`, `tzdata`

### 2. CPU-Optimized PyTorch
Replaced GPU PyTorch with CPU-only version in production builds:
```
torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Multi-Stage Docker Builds
Implemented multi-stage builds to separate build dependencies from runtime:
- Builder stage: Installs build tools and compiles dependencies
- Production stage: Only includes runtime libraries and application code

### 4. Conditional Dependency Loading
All 3D modules implement conditional imports with fallback mechanisms:
```python
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
```

### 5. Environment-Based Feature Control
Features can be toggled via environment variables:
- `ENABLE_AI_ENHANCEMENT=true/false` - Controls Stable Diffusion loading
- `ENABLE_3D_FEATURES=true/false` - Controls advanced 3D features

## Size Comparison

| Deployment Tier | Image Size | Reduction | Use Case |
|-----------------|------------|-----------|----------|
| Minimal | ~3GB | 85% | Development, basic testing |
| Production | ~5GB | 75% | Cloud Run, production deployment |
| Optimized | ~8GB | 60% | Moderate resources, better performance |
| Full | ~20GB | 0% | High-memory environments, maximum features |

## Testing Commands

### Build All Tiers
```bash
cd backend

# Build minimal
docker build -f Dockerfile.minimal -t virtualfit-minimal .

# Build production  
docker build -f Dockerfile.production -t virtualfit-production .

# Build optimized
docker build -f Dockerfile.optimized -t virtualfit-optimized .

# Build full
docker build -f Dockerfile -t virtualfit-full .

# Compare sizes
docker images | grep virtualfit
```

### Test Functionality
```bash
# Test minimal configuration
docker run -p 8000:8000 -e ENABLE_AI_ENHANCEMENT=false virtualfit-minimal

# Test production configuration
docker run -p 8000:8000 -e ENABLE_AI_ENHANCEMENT=true virtualfit-production

# Verify API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/system-status
```

## Deployment Recommendations

### Cloud Run (Google Cloud)
- **Recommended**: Production tier (~5GB)
- **Memory**: 4GB minimum
- **CPU**: 2 vCPU minimum
- **Timeout**: 900 seconds for AI processing

### Docker Compose (Local)
- **Recommended**: Optimized tier (~8GB)
- **Memory**: 8GB minimum
- **Better performance for local development**

### Kubernetes
- **Recommended**: Full tier (~20GB) for production clusters
- **Memory**: 16GB minimum per pod
- **GPU support available**

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_3D_FEATURES` | `true` | Enable MediaPipe, trimesh, PyBullet |
| `ENABLE_AI_ENHANCEMENT` | `true` | Enable Stable Diffusion AI enhancement |
| `PYTHONPATH` | `/app` | Python module search path |
| `OPENAI_API_KEY` | - | OpenAI API key for fallback processing |

## Troubleshooting

### Build Failures
- Ensure Docker has sufficient memory (8GB+)
- Check internet connectivity for package downloads
- Verify base image availability

### Runtime Issues
- Check environment variable configuration
- Verify all required dependencies are available
- Monitor memory usage during AI processing

### Performance Optimization
- Use production tier for Cloud Run deployment
- Enable AI enhancement only when needed
- Monitor container resource usage
