#!/usr/bin/env python3
"""
Server monitoring and auto-restart script
"""
import subprocess
import time
import requests
import logging
import os

def check_backend_health():
    try:
        response = requests.get("http://localhost:8001/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_frontend_health():
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        return response.status_code == 200
    except:
        return False

def restart_backend():
    subprocess.run(["pkill", "-f", "uvicorn"], check=False)
    time.sleep(2)
    subprocess.Popen([
        "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"
    ], cwd="/home/ubuntu/repos/Emergent-Virtual-Tryon/backend")

def restart_frontend():
    subprocess.run(["pkill", "-f", "craco"], check=False)
    time.sleep(2)
    subprocess.Popen([
        "npm", "start"
    ], cwd="/home/ubuntu/repos/Emergent-Virtual-Tryon/frontend")

def monitor_loop():
    while True:
        if not check_backend_health():
            logging.warning("Backend unhealthy, restarting...")
            restart_backend()
        
        if not check_frontend_health():
            logging.warning("Frontend unhealthy, restarting...")
            restart_frontend()
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor_loop()
