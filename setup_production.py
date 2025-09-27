#!/usr/bin/env python3
"""
Production setup script for VirtualFit on compute machine
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path

def setup_mongodb():
    """Setup MongoDB for production"""
    print("Setting up MongoDB...")
    
    # Create MongoDB directories
    mongodb_dir = Path("C:/mongodb")
    data_dir = mongodb_dir / "data"
    logs_dir = mongodb_dir / "logs"
    
    try:
        mongodb_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        print(f"Created MongoDB directories: {mongodb_dir}")
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False
    
    # Download MongoDB Community Server
    mongodb_url = "https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-8.0.4.zip"
    mongodb_zip = mongodb_dir / "mongodb.zip"
    
    if not (mongodb_dir / "bin" / "mongod.exe").exists():
        print("Downloading MongoDB...")
        try:
            urllib.request.urlretrieve(mongodb_url, mongodb_zip)
            print("Extracting MongoDB...")
            with zipfile.ZipFile(mongodb_zip, 'r') as zip_ref:
                zip_ref.extractall(mongodb_dir)
            
            # Move files from extracted folder to mongodb_dir
            extracted_folder = list(mongodb_dir.glob("mongodb-*"))[0]
            for item in extracted_folder.iterdir():
                if item.is_dir():
                    item.rename(mongodb_dir / item.name)
                else:
                    item.rename(mongodb_dir / item.name)
            
            extracted_folder.rmdir()
            mongodb_zip.unlink()
            print("MongoDB installed successfully")
        except Exception as e:
            print(f"Error downloading/extracting MongoDB: {e}")
            return False
    
    # Create MongoDB config file
    config_file = mongodb_dir / "mongod.conf"
    config_content = f"""
systemLog:
  destination: file
  path: {logs_dir}/mongod.log
  logAppend: true
storage:
  dbPath: {data_dir}
net:
  port: 27017
  bindIp: 127.0.0.1
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("MongoDB setup completed")
    return True

def start_mongodb():
    """Start MongoDB service"""
    print("Starting MongoDB...")
    
    mongodb_dir = Path("C:/mongodb")
    mongod_exe = mongodb_dir / "bin" / "mongod.exe"
    config_file = mongodb_dir / "mongod.conf"
    
    if not mongod_exe.exists():
        print("MongoDB not found. Run setup first.")
        return False
    
    try:
        # Start MongoDB as background process
        process = subprocess.Popen([
            str(mongod_exe),
            "--config", str(config_file)
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        print("MongoDB started successfully")
        print(f"Process ID: {process.pid}")
        return True
    except Exception as e:
        print(f"Error starting MongoDB: {e}")
        return False

def install_python_dependencies():
    """Install all required Python dependencies"""
    print("Installing Python dependencies...")
    
    requirements = [
        "fastapi==0.110.1",
        "uvicorn==0.25.0",
        "pymongo==4.5.0",
        "motor==3.3.1",
        "python-dotenv>=1.0.1",
        "pydantic>=2.6.4",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "bcrypt>=4.0.1",
        "python-multipart>=0.0.9",
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "numpy>=1.26.0",
        "opencv-python==4.8.1.78",
        "scikit-learn==1.3.2",
        "mediapipe>=0.10.11",
        "torch>=2.8.0",
        "torchvision>=0.19.0",
        "transformers>=4.45.0",
        "diffusers>=0.30.0",
        "trimesh[easy]==4.0.5",
        "pybullet==3.2.7",
        "aiohttp>=3.9.1",
        "celery>=5.3.0",
        "redis>=5.0.0",
        "flower>=2.0.1"
    ]
    
    for package in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    print("Python dependencies installation completed")

def setup_environment():
    """Setup production environment variables"""
    print("Setting up environment...")
    
    env_content = f"""# VirtualFit Production Environment
MONGO_URL=mongodb://localhost:27017
DB_NAME=virtualfit_production
CORS_ORIGINS=*
SECRET_KEY=virtualfit-production-secret-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
REDIS_URL=redis://localhost:6379/0
ENABLE_AI_ENHANCEMENT=true
ENABLE_3D_FEATURES=true
USE_COMPREHENSIVE_TRYON=true
"""
    
    env_file = Path("backend/.env")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"Environment file created: {env_file}")

def create_production_startup_script():
    """Create production startup script"""
    print("Creating production startup script...")
    
    startup_script = """@echo off
echo Starting VirtualFit Production System
echo =====================================

echo Starting MongoDB...
start "MongoDB" C:\\mongodb\\bin\\mongod.exe --config C:\\mongodb\\mongod.conf

echo Waiting for MongoDB to start...
timeout /t 5 /nobreak > nul

echo Starting VirtualFit Backend...
cd backend
python production_server.py

pause
"""
    
    with open("start_production.bat", 'w') as f:
        f.write(startup_script)
    
    print("Production startup script created: start_production.bat")

def verify_setup():
    """Verify production setup"""
    print("Verifying production setup...")
    
    checks = {
        "MongoDB directory": Path("C:/mongodb").exists(),
        "MongoDB executable": Path("C:/mongodb/bin/mongod.exe").exists(),
        "Environment file": Path("backend/.env").exists(),
        "Production server": Path("backend/production_server.py").exists(),
        "Startup script": Path("start_production.bat").exists()
    }
    
    all_good = True
    for check, status in checks.items():
        status_text = "✓" if status else "✗"
        print(f"{status_text} {check}")
        if not status:
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("VirtualFit Production Setup")
    print("=" * 30)
    
    steps = [
        ("MongoDB Setup", setup_mongodb),
        ("Python Dependencies", install_python_dependencies),
        ("Environment Configuration", setup_environment),
        ("Production Scripts", create_production_startup_script),
        ("Verification", verify_setup)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            result = step_func()
            if result is False:
                print(f"❌ {step_name} failed")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    print("\n🎉 Production setup completed successfully!")
    print("\nNext steps:")
    print("1. Update OPENAI_API_KEY in backend/.env")
    print("2. Run: start_production.bat")
    print("3. Access API at: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    main()