#!/bin/bash

# Production Deployment Script for VirtualFit
# Prepares compute engine with all dependencies for Google Vertex AI + FAHN + PrecisePoseDetector

set -e

echo "🚀 Starting VirtualFit Production Deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for MediaPipe and OpenCV
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libopenexr-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    ffmpeg \
    git \
    curl \
    wget

# Pull latest code from GitHub
echo "📥 Pulling latest code from GitHub..."
cd /opt/virtualfit || { echo "❌ VirtualFit directory not found"; exit 1; }
git pull origin main

# Create/activate virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies in correct order
echo "📚 Installing Python dependencies..."

# Install NumPy first (specific version for compatibility)
pip install "numpy>=1.26.0,<2.0"

# Install OpenCV with specific version
pip install opencv-python==4.8.1.78

# Install MediaPipe
pip install mediapipe>=0.10.11

# Install core ML dependencies
pip install torch>=2.8.0 torchvision>=0.19.0 torchaudio>=2.8.0

# Install remaining requirements
pip install -r backend/requirements.txt

# Set up environment variables
echo "⚙️ Setting up environment variables..."
cd backend

# Create production .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating production .env file..."
    cat > .env << EOF
# Production Environment Configuration
MONGO_URL="mongodb://localhost:27017"
DB_NAME="virtualfit_production"
CORS_ORIGINS="*"

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# OpenAI Configuration (fallback)
OPENAI_API_KEY="your-openai-api-key"

# Feature Flags
USE_COMPREHENSIVE_TRYON=true
ENABLE_AI_ENHANCEMENT=true
ENABLE_3D_FEATURES=true

# Performance Settings
MAX_WORKERS=4
CACHE_SIZE=1000
GPU_MEMORY_FRACTION=0.8
EOF
    echo "⚠️  Please update .env with your actual credentials"
fi

# Install MongoDB if not present
if ! command -v mongod &> /dev/null; then
    echo "🗄️ Installing MongoDB..."
    wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    sudo apt update
    sudo apt install -y mongodb-org
    sudo systemctl start mongod
    sudo systemctl enable mongod
fi

# Test critical imports
echo "🧪 Testing critical imports..."
python3 -c "
import cv2
import mediapipe as mp
import numpy as np
import torch
import transformers
from src.core.precise_pose_detector import PrecisePoseDetector
from src.core.customer_image_analyzer import CustomerImageAnalyzer
print('✅ All critical imports successful')
"

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/virtualfit.service > /dev/null << EOF
[Unit]
Description=VirtualFit Backend Service
After=network.target mongodb.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/virtualfit/backend
Environment=PATH=/opt/virtualfit/venv/bin
ExecStart=/opt/virtualfit/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
sudo chown -R www-data:www-data /opt/virtualfit
sudo chmod +x /opt/virtualfit/backend/server.py

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable virtualfit
sudo systemctl restart virtualfit

# Test service
echo "🔍 Testing service..."
sleep 5
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ VirtualFit service is running successfully"
else
    echo "❌ Service test failed, checking logs..."
    sudo journalctl -u virtualfit --no-pager -n 20
fi

echo "🎉 Production deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env with your actual credentials"
echo "2. Configure Google Cloud service account"
echo "3. Test virtual try-on functionality"
echo "4. Monitor logs: sudo journalctl -u virtualfit -f"