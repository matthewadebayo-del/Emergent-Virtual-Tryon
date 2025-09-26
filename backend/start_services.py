#!/usr/bin/env python3
"""
Production service starter for VirtualFit dual-mode system
"""
import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def start_redis():
    """Start Redis server"""
    print("Starting Redis server...")
    try:
        subprocess.Popen(["redis-server", "--daemonize", "yes"])
        time.sleep(2)
        print("✓ Redis started")
        return True
    except FileNotFoundError:
        print("✗ Redis not found. Install with: apt-get install redis-server")
        return False

def start_celery_worker():
    """Start Celery worker"""
    print("Starting Celery worker...")
    cmd = [
        sys.executable, "-m", "celery", 
        "-A", "src.workers.celery_app", 
        "worker", 
        "--loglevel=info",
        "--concurrency=2"
    ]
    return subprocess.Popen(cmd)

def start_celery_beat():
    """Start Celery beat scheduler"""
    print("Starting Celery beat...")
    cmd = [
        sys.executable, "-m", "celery",
        "-A", "src.workers.celery_app",
        "beat",
        "--loglevel=info"
    ]
    return subprocess.Popen(cmd)

def start_flower():
    """Start Flower monitoring"""
    print("Starting Flower monitoring...")
    cmd = [
        sys.executable, "-m", "celery",
        "-A", "src.workers.celery_app",
        "flower",
        "--port=5555"
    ]
    return subprocess.Popen(cmd)

def start_fastapi():
    """Start FastAPI server"""
    print("Starting FastAPI server...")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "production_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    return subprocess.Popen(cmd)

def main():
    """Start all services"""
    print("🚀 Starting VirtualFit Production Services")
    print("=" * 50)
    
    processes = []
    
    try:
        # Start Redis
        if not start_redis():
            sys.exit(1)
        
        # Start Celery worker
        worker_process = start_celery_worker()
        processes.append(worker_process)
        time.sleep(3)
        
        # Start Celery beat
        beat_process = start_celery_beat()
        processes.append(beat_process)
        time.sleep(2)
        
        # Start Flower
        flower_process = start_flower()
        processes.append(flower_process)
        time.sleep(2)
        
        # Start FastAPI
        api_process = start_fastapi()
        processes.append(api_process)
        
        print("\n✅ All services started successfully!")
        print("\n📊 Service URLs:")
        print("  • API Server: http://localhost:8000")
        print("  • API Docs: http://localhost:8000/docs")
        print("  • Flower Monitor: http://localhost:5555")
        print("  • Integration API: http://localhost:8000/api/v1")
        
        print("\n🔧 Integration Endpoints:")
        print("  • POST /api/v1/tryon/process - Async processing")
        print("  • POST /api/v1/tryon/sync - Sync processing")
        print("  • GET /api/v1/tryon/status/{job_id} - Check status")
        print("  • GET /api/v1/tryon/result/{job_id} - Get result")
        
        print("\n⏹️  Press Ctrl+C to stop all services")
        
        # Wait for interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for process in processes:
            process.terminate()
        
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if needed
        for process in processes:
            if process.poll() is None:
                process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()