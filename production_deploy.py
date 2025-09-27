#!/usr/bin/env python3
"""
Production deployment script for VirtualFit on compute machine
"""
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

class ProductionDeployer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.backend_dir = self.base_dir / "backend"
        self.frontend_dir = self.base_dir / "frontend"
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("Checking prerequisites...")
        
        checks = {
            "Python": sys.version_info >= (3, 8),
            "Backend directory": self.backend_dir.exists(),
            "Frontend directory": self.frontend_dir.exists(),
            "Production server": (self.backend_dir / "production_server.py").exists(),
            "Environment file": (self.backend_dir / ".env").exists()
        }
        
        all_good = True
        for check, status in checks.items():
            status_text = "✓" if status else "✗"
            print(f"  {status_text} {check}")
            if not status:
                all_good = False
        
        return all_good
    
    def start_mongodb(self):
        """Start MongoDB service"""
        print("Starting MongoDB...")
        
        mongodb_exe = Path("C:/mongodb/bin/mongod.exe")
        config_file = Path("C:/mongodb/mongod.conf")
        
        if not mongodb_exe.exists():
            print("  ✗ MongoDB not found. Run setup_production.py first.")
            return False
        
        try:
            # Check if MongoDB is already running
            try:
                response = requests.get("http://localhost:27017", timeout=2)
                print("  ✓ MongoDB already running")
                return True
            except:
                pass
            
            # Start MongoDB
            process = subprocess.Popen([
                str(mongodb_exe),
                "--config", str(config_file)
            ], creationflags=subprocess.CREATE_NEW_CONSOLE)
            
            # Wait for MongoDB to start
            for i in range(10):
                try:
                    import pymongo
                    client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1000)
                    client.server_info()
                    print("  ✓ MongoDB started successfully")
                    return True
                except:
                    time.sleep(2)
            
            print("  ✗ MongoDB failed to start")
            return False
            
        except Exception as e:
            print(f"  ✗ Error starting MongoDB: {e}")
            return False
    
    def start_backend(self):
        """Start backend server"""
        print("Starting backend server...")
        
        try:
            # Change to backend directory
            os.chdir(self.backend_dir)
            
            # Start production server
            process = subprocess.Popen([
                sys.executable, "production_server.py"
            ])
            
            # Wait for server to start
            for i in range(30):
                try:
                    response = requests.get("http://localhost:8000/health", timeout=2)
                    if response.status_code == 200:
                        print("  ✓ Backend server started successfully")
                        return process
                except:
                    time.sleep(2)
            
            print("  ✗ Backend server failed to start")
            return None
            
        except Exception as e:
            print(f"  ✗ Error starting backend: {e}")
            return None
    
    def start_frontend(self):
        """Start frontend server"""
        print("Starting frontend server...")
        
        try:
            # Check if node_modules exists
            node_modules = self.frontend_dir / "node_modules"
            if not node_modules.exists():
                print("  Installing frontend dependencies...")
                subprocess.run(["yarn", "install"], cwd=self.frontend_dir, check=True)
            
            # Start frontend
            process = subprocess.Popen([
                "yarn", "start"
            ], cwd=self.frontend_dir)
            
            # Wait for frontend to start
            for i in range(60):
                try:
                    response = requests.get("http://localhost:3000", timeout=2)
                    if response.status_code == 200:
                        print("  ✓ Frontend server started successfully")
                        return process
                except:
                    time.sleep(2)
            
            print("  ✗ Frontend server failed to start")
            return None
            
        except Exception as e:
            print(f"  ✗ Error starting frontend: {e}")
            return None
    
    def verify_deployment(self):
        """Verify that all services are running"""
        print("Verifying deployment...")
        
        services = {
            "Backend API": "http://localhost:8000/health",
            "Frontend": "http://localhost:3000",
            "API Docs": "http://localhost:8000/docs"
        }
        
        all_good = True
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✓ {service}: OK")
                else:
                    print(f"  ✗ {service}: HTTP {response.status_code}")
                    all_good = False
            except Exception as e:
                print(f"  ✗ {service}: {e}")
                all_good = False
        
        return all_good
    
    def deploy(self):
        """Run complete deployment"""
        print("🚀 VirtualFit Production Deployment")
        print("=" * 40)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Run setup_production.py first.")
            return False
        
        # Start services
        if not self.start_mongodb():
            print("❌ MongoDB startup failed")
            return False
        
        backend_process = self.start_backend()
        if not backend_process:
            print("❌ Backend startup failed")
            return False
        
        frontend_process = self.start_frontend()
        if not frontend_process:
            print("❌ Frontend startup failed")
            return False
        
        # Verify deployment
        time.sleep(5)  # Give services time to fully start
        if not self.verify_deployment():
            print("❌ Deployment verification failed")
            return False
        
        print("\n🎉 Production deployment successful!")
        print("\n📊 Service URLs:")
        print("  • Frontend: http://localhost:3000")
        print("  • Backend API: http://localhost:8000")
        print("  • API Documentation: http://localhost:8000/docs")
        print("  • Health Check: http://localhost:8000/health")
        print("  • Integration API: http://localhost:8000/api/v1")
        
        print("\n⚡ System Status:")
        try:
            health_response = requests.get("http://localhost:8000/health")
            health_data = health_response.json()
            print(f"  • Status: {health_data.get('status', 'unknown')}")
            components = health_data.get('components', {})
            for component, status in components.items():
                status_text = "✓" if status else "✗"
                print(f"  • {component}: {status_text}")
        except:
            print("  • Could not retrieve system status")
        
        print("\n🔧 Management:")
        print("  • Press Ctrl+C to stop all services")
        print("  • Logs are available in backend/logs/")
        
        try:
            # Keep processes running
            print("\n⏳ Services running... Press Ctrl+C to stop")
            while True:
                time.sleep(10)
                # Basic health check
                try:
                    requests.get("http://localhost:8000/health", timeout=2)
                except:
                    print("⚠️  Backend health check failed")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            try:
                backend_process.terminate()
                frontend_process.terminate()
                print("✅ All services stopped")
            except:
                pass
        
        return True

def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    success = deployer.deploy()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()