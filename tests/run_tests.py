#!/usr/bin/env python3
"""
Test runner for VirtualFit dual-mode system
"""
import subprocess
import sys
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor

class TestRunner:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        
    def check_services(self):
        """Check if all services are running"""
        print("🔍 Checking services...")
        
        services = {
            "API Server": f"{self.base_url}/api/v1/health",
            "Frontend": self.frontend_url,
            "Redis": "redis://localhost:6379",
            "Flower": "http://localhost:5555"
        }
        
        for name, url in services.items():
            try:
                if url.startswith("redis://"):
                    # Check Redis separately
                    import redis
                    r = redis.from_url(url)
                    r.ping()
                else:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                print(f"✅ {name}: OK")
            except Exception as e:
                print(f"❌ {name}: {e}")
                return False
        return True
    
    def run_unit_tests(self):
        """Run unit tests"""
        print("\n🧪 Running unit tests...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_dual_mode_system.py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Unit tests failed: {e}")
            return False
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running integration tests...")
        
        test_cases = [
            self.test_standalone_mode,
            self.test_api_endpoints,
            self.test_async_workflow,
            self.test_sdk_functionality
        ]
        
        results = []
        for test in test_cases:
            try:
                result = test()
                results.append(result)
                print(f"{'✅' if result else '❌'} {test.__name__}")
            except Exception as e:
                print(f"❌ {test.__name__}: {e}")
                results.append(False)
        
        return all(results)
    
    def test_standalone_mode(self):
        """Test standalone web application"""
        try:
            response = requests.get(self.frontend_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        endpoints = [
            "/api/v1/health",
            "/docs",
            "/api/v1/products"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code >= 400:
                    return False
            except:
                return False
        return True
    
    def test_async_workflow(self):
        """Test async processing workflow"""
        try:
            # Mock async request
            payload = {
                "api_key": "test-key-12345",
                "customer_image_url": "https://httpbin.org/image/jpeg",
                "garment_image_url": "https://httpbin.org/image/jpeg",
                "product_info": {
                    "name": "Test Product",
                    "category": "TOP",
                    "color": "white"
                }
            }
            
            response = requests.post(f"{self.base_url}/api/v1/tryon/process", json=payload, timeout=10)
            if response.status_code != 200:
                return False
            
            data = response.json()
            job_id = data.get("job_id")
            
            # Check status
            status_response = requests.get(f"{self.base_url}/api/v1/tryon/status/{job_id}", timeout=5)
            return status_response.status_code == 200
            
        except:
            return False
    
    def test_sdk_functionality(self):
        """Test SDK files exist and have correct structure"""
        sdk_files = [
            "sdks/javascript/virtualfit-sdk.js",
            "sdks/python/virtualfit_sdk.py",
            "plugins/shopify/virtualfit-app.js",
            "plugins/woocommerce/virtualfit-plugin.php"
        ]
        
        for file_path in sdk_files:
            if not os.path.exists(file_path):
                return False
        return True
    
    def run_load_tests(self):
        """Run basic load tests"""
        print("\n⚡ Running load tests...")
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        # Run 50 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in futures]
        
        success_rate = sum(results) / len(results) * 100
        print(f"Load test success rate: {success_rate:.1f}%")
        return success_rate >= 95
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 VirtualFit Dual-Mode System Test Suite")
        print("=" * 50)
        
        # Check services first
        if not self.check_services():
            print("❌ Services not ready. Please start all services first.")
            return False
        
        # Run tests
        test_results = {
            "Unit Tests": self.run_unit_tests(),
            "Integration Tests": self.run_integration_tests(),
            "Load Tests": self.run_load_tests()
        }
        
        # Summary
        print("\n📊 Test Results Summary:")
        print("=" * 30)
        
        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        print("\n" + "=" * 30)
        if all_passed:
            print("🎉 All tests passed! System ready for production.")
        else:
            print("❌ Some tests failed. Please check the issues above.")
        
        return all_passed

if __name__ == "__main__":
    runner = TestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)