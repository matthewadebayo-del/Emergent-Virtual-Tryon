#!/usr/bin/env python3
"""
Simple production starter for VirtualFit
"""
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def start_mongodb():
    """Start MongoDB"""
    print("Starting MongoDB...")
    
    mongodb_exe = Path("C:/mongodb/bin/mongod.exe")
    config_file = Path("C:/mongodb/mongod.conf")
    
    if not mongodb_exe.exists():
        print("MongoDB not found at C:/mongodb/bin/mongod.exe")
        return False
    
    try:
        # Start MongoDB in background
        process = subprocess.Popen([
            str(mongodb_exe),
            "--config", str(config_file)
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        print(f"MongoDB started with PID: {process.pid}")
        
        # Wait for MongoDB to be ready
        for i in range(10):
            try:
                import pymongo
                client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1000)
                client.server_info()
                print("MongoDB is ready")
                return True
            except:
                time.sleep(2)
        
        print("MongoDB failed to start properly")
        return False
        
    except Exception as e:
        print(f"Error starting MongoDB: {e}")
        return False

def start_backend():
    """Start backend server"""
    print("Starting VirtualFit backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("Backend directory not found")
        return None
    
    try:
        # Change to backend directory and start server
        process = subprocess.Popen([
            sys.executable, "production_server.py"
        ], cwd=backend_dir)
        
        print(f"Backend started with PID: {process.pid}")
        
        # Wait for backend to be ready
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("Backend is ready")
                    return process
            except:
                time.sleep(2)
        
        print("Backend failed to start properly")
        return None
        
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def main():
    """Main function"""
    print("VirtualFit Production Deployment")
    print("=" * 40)
    
    # Start MongoDB
    if not start_mongodb():
        print("Failed to start MongoDB")
        return False
    
    # Start Backend
    backend_process = start_backend()
    if not backend_process:
        print("Failed to start backend")
        return False
    
    # Show status
    print("\nProduction services started successfully!")
    print("\nService URLs:")
    print("  Backend API: http://localhost:8000")
    print("  API Documentation: http://localhost:8000/docs")
    print("  Health Check: http://localhost:8000/health")
    print("  Integration API: http://localhost:8000/api/v1")
    
    # Test health
    try:
        response = requests.get("http://localhost:8000/health")
        health_data = response.json()
        print(f"\nSystem Status: {health_data.get('status', 'unknown')}")
        
        components = health_data.get('components', {})
        for component, status in components.items():
            status_text = "OK" if status else "FAIL"
            print(f"  {component}: {status_text}")
            
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\nPress Ctrl+C to stop services")
    
    try:
        # Keep running
        while True:
            time.sleep(10)
            # Basic health check
            try:
                requests.get("http://localhost:8000/health", timeout=2)
            except:
                print("Warning: Backend health check failed")
    
    except KeyboardInterrupt:
        print("\nStopping services...")
        try:
            backend_process.terminate()
            backend_process.wait(timeout=5)
            print("Backend stopped")
        except:
            print("Force killing backend...")
            backend_process.kill()
        
        print("All services stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)