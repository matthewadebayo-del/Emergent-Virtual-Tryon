#!/usr/bin/env python3
"""
Install Production Dependencies for VirtualFit 3D Pipeline
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing Production Dependencies for VirtualFit 3D Pipeline")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment. Proceeding anyway...")
    
    # Install AI/ML dependencies
    ai_packages = [
        "torch==2.0.1+cpu",
        "torchvision==0.15.2+cpu", 
        "transformers==4.30.0",
        "diffusers==0.18.0",
        "accelerate==0.20.0"
    ]
    
    # Install 3D processing dependencies
    mesh_packages = [
        "trimesh==3.22.0",
        "scipy==1.10.0",
        "scikit-image==0.20.0",
        "networkx==3.1"
    ]
    
    # Install physics simulation (already working)
    physics_packages = [
        "pybullet==3.2.5"
    ]
    
    # Install additional dependencies
    extra_packages = [
        "mediapipe==0.10.0",
        "open3d==0.17.0"
    ]
    
    all_packages = ai_packages + mesh_packages + physics_packages + extra_packages
    
    # Install packages
    for package in all_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Install PyTorch with CPU support (lighter version)
    print("\n🔄 Installing PyTorch CPU version...")
    torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    run_command(torch_cmd, "Installing PyTorch CPU")
    
    print("\n✅ Production dependencies installation completed!")
    print("\n📋 Next steps:")
    print("1. Restart the server: pkill uvicorn && nohup uvicorn production_server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &")
    print("2. Check health: curl http://localhost:8000/health")
    print("3. Verify features: curl http://localhost:8000/")

if __name__ == "__main__":
    main()