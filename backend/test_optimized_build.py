#!/usr/bin/env python3
"""
Test script to verify optimized Docker build functionality
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def test_dependency_imports():
    """Test that all required dependencies can be imported"""
    print("🔍 Testing dependency imports...")
    
    required_modules = [
        'fastapi',
        'uvicorn', 
        'pymongo',
        'mediapipe',
        'cv2',
        'trimesh',
        'pybullet',
        'torch',
        'diffusers',
        'PIL',
        'numpy',
        'scipy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {failed_imports}")
        return False
    
    print("✅ All required dependencies imported successfully")
    return True

def test_3d_modules():
    """Test that 3D virtual try-on modules work"""
    print("\n🔍 Testing 3D virtual try-on modules...")
    
    try:
        sys.path.append('src')
        from src.core.body_reconstruction import BodyReconstructor
        from src.core.garment_fitting import GarmentFitter
        from src.core.rendering import PhotorealisticRenderer
        from src.core.ai_enhancement import AIEnhancer
        
        body_reconstructor = BodyReconstructor()
        garment_fitter = GarmentFitter()
        renderer = PhotorealisticRenderer()
        ai_enhancer = AIEnhancer()
        
        print("✅ All 3D modules initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ 3D module test failed: {e}")
        return False

def test_server_startup():
    """Test that the server can start up"""
    print("\n🔍 Testing server startup...")
    
    try:
        import server
        print("✅ Server module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

def main():
    """Run all optimization tests"""
    print("🚀 Testing optimized Docker build functionality\n")
    
    tests = [
        test_dependency_imports,
        test_3d_modules,
        test_server_startup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimized build is ready for deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the optimization.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
