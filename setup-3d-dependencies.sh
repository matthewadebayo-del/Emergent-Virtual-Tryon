#!/bin/bash

# Setup Full 3D Dependencies for VirtualFit
# Run this script on the Compute Engine instance

echo "🔧 Setting up Full 3D Virtual Try-On Dependencies..."

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install build essentials
sudo apt-get install -y build-essential cmake pkg-config git wget curl

# Install Python 3.9 and development tools
sudo apt-get install -y python3.9 python3.9-dev python3-pip python3.9-venv

# Install OpenCV and computer vision libraries
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install MediaPipe system dependencies
sudo apt-get install -y \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# Install PyBullet physics simulation dependencies
sudo apt-get install -y \
    libbullet-dev \
    libbullet-extras-dev \
    libode-dev

# Install Blender for 3D rendering
sudo apt-get install -y blender xvfb

# Install additional 3D processing libraries
sudo apt-get install -y \
    libfreeimage3 \
    libfreeimage-dev \
    libassimp5 \
    libassimp-dev \
    libpcl-dev

# Create Python virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core ML/AI stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install 3D processing stack
pip install \
    mediapipe==0.10.7 \
    trimesh==4.0.5 \
    pybullet==3.2.5 \
    open3d==0.18.0

# Install Stable Diffusion stack
pip install \
    diffusers==0.24.0 \
    transformers==4.35.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors==0.4.1

# Install web framework
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6

# Install remaining dependencies
pip install -r backend/requirements-full-3d.txt

# Test 3D stack installation
echo "🧪 Testing 3D stack installation..."

python3 -c "
import mediapipe as mp
print('✅ MediaPipe:', mp.__version__)

import trimesh
print('✅ Trimesh:', trimesh.__version__)

import pybullet as p
print('✅ PyBullet: Available')

import torch
print('✅ PyTorch:', torch.__version__)

import diffusers
print('✅ Diffusers:', diffusers.__version__)

print('🎉 All 3D dependencies installed successfully!')
"

echo "✅ Full 3D stack setup complete!"
echo "🚀 Ready to run VirtualFit with complete 3D virtual try-on pipeline"