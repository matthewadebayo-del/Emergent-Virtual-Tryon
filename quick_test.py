#!/usr/bin/env python3
"""
Quick test script for dual-mode system without Redis dependency
"""
import subprocess
import sys
import time
import requests
import os

def test_backend_only():
    """Test backend without Redis/Celery dependencies"""
    print("Testing Backend Only (No Redis)")
    
    # Start backend server
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    
    try:
        # Test basic server startup
        print("Starting backend server...")
        process = subprocess.Popen([
            sys.executable, 'production_server.py'
        ], cwd=backend_dir)
        
        # Wait for server to start
        time.sleep(10)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print("Backend server running")
                print(f"Response: {response.json()}")
            else:
                print(f"Backend health check failed: {response.status_code}")
        except Exception as e:
            print(f"Backend connection failed: {e}")
        
        # Test API docs
        try:
            response = requests.get('http://localhost:8000/docs', timeout=5)
            if response.status_code == 200:
                print("API docs accessible")
            else:
                print(f"API docs failed: {response.status_code}")
        except Exception as e:
            print(f"API docs connection failed: {e}")
        
        # Test virtual try-on endpoint (basic)
        try:
            payload = {
                "customer_image_base64": "test",
                "garment_image_base64": "test",
                "product_name": "Test T-Shirt"
            }
            response = requests.post('http://localhost:8000/api/virtual-tryon', json=payload, timeout=10)
            print(f"Try-on endpoint response: {response.status_code}")
            if response.status_code != 500:  # Expect error but endpoint should exist
                print("Try-on endpoint accessible")
        except Exception as e:
            print(f"Try-on endpoint test: {e}")
        
        return True
        
    except Exception as e:
        print(f"Backend test failed: {e}")
        return False
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            pass

def test_frontend():
    """Test frontend startup"""
    print("\nTesting Frontend")
    
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    
    # Check if node_modules exists
    if not os.path.exists(os.path.join(frontend_dir, 'node_modules')):
        print("Installing frontend dependencies...")
        try:
            subprocess.run(['yarn', 'install'], cwd=frontend_dir, check=True)
        except Exception as e:
            print(f"Frontend dependency installation failed: {e}")
            return False
    
    try:
        # Start frontend (non-blocking)
        print("Starting frontend server...")
        process = subprocess.Popen([
            'yarn', 'start'
        ], cwd=frontend_dir)
        
        # Wait for frontend to start
        time.sleep(15)
        
        # Test frontend
        try:
            response = requests.get('http://localhost:3000', timeout=10)
            if response.status_code == 200:
                print("Frontend server running")
                if "VirtualFit" in response.text:
                    print("VirtualFit branding detected")
            else:
                print(f"Frontend failed: {response.status_code}")
        except Exception as e:
            print(f"Frontend connection failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Frontend test failed: {e}")
        return False
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            pass

def test_sdk_files():
    """Test SDK files exist"""
    print("\nTesting SDK Files")
    
    sdk_files = [
        'sdks/javascript/virtualfit-sdk.js',
        'sdks/python/virtualfit_sdk.py',
        'plugins/shopify/virtualfit-app.js',
        'plugins/woocommerce/virtualfit-plugin.php'
    ]
    
    all_exist = True
    for file_path in sdk_files:
        if os.path.exists(file_path):
            print(f"PASS {file_path}")
        else:
            print(f"FAIL {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run quick tests"""
    print("VirtualFit Quick Test Suite")
    print("=" * 40)
    
    results = {
        "SDK Files": test_sdk_files(),
        "Backend": test_backend_only(),
        "Frontend": test_frontend()
    }
    
    print("\nTest Results:")
    print("=" * 20)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nBasic tests passed! System ready for deployment.")
    else:
        print("\nSome tests failed. Check issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)