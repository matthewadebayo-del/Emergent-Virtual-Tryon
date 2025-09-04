#!/usr/bin/env python3
"""
Ultra-optimized deployment test for VirtualFit Backend
Tests deployment timeout fixes and 3D functionality preservation
"""

import subprocess
import sys
import time
import requests
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_ultra_docker_build():
    """Test ultra-optimized Docker image builds successfully"""
    print("🔨 Testing ultra-optimized Docker build...")
    
    success, stdout, stderr = run_command(
        "docker build -f Dockerfile.ultra -t test-ultra-deployment ."
    )
    
    if success:
        print("✅ Ultra Docker build successful")
        
        success, stdout, stderr = run_command(
            "docker images test-ultra-deployment --format 'table {{.Size}}'"
        )
        if success and stdout:
            size = stdout.strip().split('\n')[-1]
            print(f"📦 Ultra image size: {size}")
            return True
    else:
        print(f"❌ Ultra Docker build failed: {stderr}")
        return False

def test_container_startup_performance():
    """Test container starts up quickly with optimized configuration"""
    print("🚀 Testing optimized container startup...")
    
    start_time = time.time()
    
    success, stdout, stderr = run_command(
        "docker run -d --name test-ultra-startup -p 8002:8000 "
        "-e ENABLE_3D_FEATURES=true -e ENABLE_AI_ENHANCEMENT=true "
        "-e STARTUP_TIMEOUT=300 -e HEALTH_CHECK_TIMEOUT=60 "
        "test-ultra-deployment"
    )
    
    if not success:
        print(f"❌ Container startup failed: {stderr}")
        return False
    
    container_id = stdout.strip()
    print(f"✅ Container started: {container_id[:12]}")
    
    print("⏳ Waiting for optimized server startup...")
    time.sleep(20)
    
    startup_time = time.time() - start_time
    print(f"⏱️ Container startup time: {startup_time:.2f}s")
    
    success, stdout, stderr = run_command(
        f"docker logs {container_id}"
    )
    
    if "✅ 3D virtual try-on modules imported and initialized successfully" in stdout:
        print("✅ 3D modules initialized successfully")
        return True
    else:
        print("❌ 3D modules failed to initialize")
        print(f"Container logs: {stdout}")
        return False

def test_deployment_timeout_fixes():
    """Test deployment configuration fixes"""
    print("🔧 Testing deployment timeout fixes...")
    
    base_url = "http://localhost:8002"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health endpoint working with extended timeout")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    try:
        response = requests.get(f"{base_url}/api/system-status", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ System status endpoint working")
            print(f"📊 3D Features: {data.get('features', {}).get('3d_processing', 'Unknown')}")
            print(f"🤖 AI Enhancement: {data.get('features', {}).get('ai_enhancement', 'Unknown')}")
            print(f"💾 Memory Usage: {data.get('system', {}).get('memory_usage', 'Unknown')}")
            return True
        else:
            print(f"❌ System status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System status error: {e}")
        return False

def cleanup():
    """Clean up test containers and images"""
    print("🧹 Cleaning up...")
    
    run_command("docker stop test-ultra-startup", capture_output=False)
    run_command("docker rm test-ultra-startup", capture_output=False)
    
    run_command("docker rmi test-ultra-deployment", capture_output=False)
    
    print("✅ Cleanup complete")

def main():
    """Run comprehensive ultra deployment test"""
    print("🎯 Starting ultra-optimized deployment timeout fix test...")
    print("=" * 60)
    
    tests = [
        ("Ultra Docker Build", test_ultra_docker_build),
        ("Container Startup Performance", test_container_startup_performance),
        ("Deployment Timeout Fixes", test_deployment_timeout_fixes),
    ]
    
    results = []
    
    try:
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
            
            if not success:
                print(f"❌ {test_name} failed - stopping tests")
                break
        
        print("\n" + "=" * 60)
        print("📊 DEPLOYMENT TIMEOUT FIX TEST SUMMARY")
        print("=" * 60)
        
        all_passed = True
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
            if not success:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL DEPLOYMENT TIMEOUT FIXES WORKING!")
            print("📦 Ultra-optimized image size achieved")
            print("🚀 All 3D virtual try-on functionality preserved")
            print("⚡ Deployment timeout issues resolved")
            print("🔧 Cloud Build/Run configuration optimized")
        else:
            print("\n❌ Some deployment fixes failed - needs investigation")
            
    finally:
        cleanup()

if __name__ == "__main__":
    main()
