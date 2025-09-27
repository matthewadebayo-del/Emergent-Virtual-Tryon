#!/usr/bin/env python3
"""
Quick production readiness test
"""
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def test_mongodb():
    """Test MongoDB installation and startup"""
    print("Testing MongoDB...")
    
    mongodb_exe = Path("C:/mongodb/bin/mongod.exe")
    if not mongodb_exe.exists():
        return False, "MongoDB executable not found"
    
    # Try to start MongoDB briefly
    try:
        process = subprocess.Popen([
            str(mongodb_exe),
            "--config", "C:/mongodb/mongod.conf"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it 5 seconds to start
        time.sleep(5)
        
        # Test connection
        try:
            import pymongo
            client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
            client.server_info()
            process.terminate()
            return True, "MongoDB working"
        except Exception as e:
            process.terminate()
            return False, f"MongoDB connection failed: {e}"
            
    except Exception as e:
        return False, f"MongoDB startup failed: {e}"

def test_backend():
    """Test backend startup"""
    print("Testing Backend...")
    
    backend_dir = Path("backend")
    if not (backend_dir / "production_server.py").exists():
        return False, "Backend server not found"
    
    # Test import of key modules
    try:
        sys.path.insert(0, str(backend_dir))
        
        # Test core imports
        import production_server
        return True, "Backend imports successful"
        
    except Exception as e:
        return False, f"Backend import failed: {e}"

def test_dependencies():
    """Test key dependencies"""
    print("Testing Dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pymongo', 'motor', 'torch', 
        'torchvision', 'transformers', 'diffusers', 'pillow',
        'numpy', 'opencv-python', 'aiohttp', 'celery', 'redis'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {missing}"
    else:
        return True, "All dependencies available"

def test_environment():
    """Test environment configuration"""
    print("Testing Environment...")
    
    env_file = Path("backend/.env")
    if not env_file.exists():
        return False, "Environment file not found"
    
    # Check key environment variables
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = ['MONGO_URL', 'DB_NAME', 'SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
    
    if missing_vars:
        return False, f"Missing environment variables: {missing_vars}"
    else:
        return True, "Environment configured"

def test_files():
    """Test required files exist"""
    print("Testing Files...")
    
    required_files = [
        "backend/production_server.py",
        "backend/.env",
        "C:/mongodb/bin/mongod.exe",
        "C:/mongodb/mongod.conf",
        "start_production.bat"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        return False, f"Missing files: {missing_files}"
    else:
        return True, "All required files present"

def main():
    """Run all tests"""
    print("VirtualFit Production Readiness Test")
    print("=" * 40)
    
    tests = [
        ("Files", test_files),
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("Backend", test_backend),
        ("MongoDB", test_mongodb)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results[test_name] = (success, message)
            
            status = "PASS" if success else "FAIL"
            print(f"{test_name}: {status} - {message}")
            
            if not success:
                all_passed = False
                
        except Exception as e:
            results[test_name] = (False, str(e))
            print(f"{test_name}: FAIL - {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    print("PRODUCTION READINESS SUMMARY")
    print("=" * 40)
    
    for test_name, (success, message) in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    if all_passed:
        print("\n🎉 PRODUCTION READY!")
        print("\nTo start production:")
        print("1. Run: python start_production_simple.py")
        print("2. Or use: start_production.bat")
        print("3. Access API at: http://localhost:8000")
    else:
        print("\n❌ NOT PRODUCTION READY")
        print("Fix the failed tests above before deployment")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)