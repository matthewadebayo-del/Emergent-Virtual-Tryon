#!/usr/bin/env python3
"""
Simple backend starter without MongoDB dependency for testing
"""
import subprocess
import sys
import os
import time

def start_backend():
    """Start backend server"""
    print("Starting VirtualFit Backend Server...")
    print("=" * 40)
    
    # Set environment for in-memory mode
    os.environ["USE_MEMORY_DB"] = "true"
    
    try:
        # Start the production server
        process = subprocess.Popen([
            sys.executable, "production_server.py"
        ], cwd="backend")
        
        print("Backend server starting...")
        print("- API Server: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs")
        print("- Health Check: http://localhost:8000/health")
        print("- Integration API: http://localhost:8000/api/v1")
        print("\nPress Ctrl+C to stop the server")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping backend server...")
        process.terminate()
        process.wait()
        print("Backend server stopped")
    except Exception as e:
        print(f"Error starting backend: {e}")

if __name__ == "__main__":
    start_backend()