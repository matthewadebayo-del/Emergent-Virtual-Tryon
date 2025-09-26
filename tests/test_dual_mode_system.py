"""
Comprehensive test suite for dual-mode virtual try-on system
"""
import pytest
import requests
import asyncio
import time
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from src.integrations.tryon_orchestrator import VirtualTryOnOrchestrator
from src.workers.tryon_tasks import process_tryon_async
from src.utils.redis_client import redis_client

class TestDualModeSystem:
    
    @pytest.fixture
    def api_base_url(self):
        return "http://localhost:8000/api/v1"
    
    @pytest.fixture
    def test_images(self):
        return {
            "customer": "https://example.com/customer.jpg",
            "garment": "https://example.com/tshirt.jpg"
        }
    
    @pytest.fixture
    def product_info(self):
        return {
            "name": "Classic White T-Shirt",
            "category": "TOP",
            "color": "white",
            "size": "M"
        }

    def test_standalone_mode(self):
        """Test original standalone web application"""
        response = requests.get("http://localhost:3000")
        assert response.status_code == 200
        assert "VirtualFit" in response.text

    def test_api_health_check(self, api_base_url):
        """Test API health endpoint"""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "VirtualFit Integration API" in data["service"]

    def test_sync_processing(self, api_base_url, test_images, product_info):
        """Test synchronous processing endpoint"""
        payload = {
            "api_key": "test-key-12345",
            "customer_image_url": test_images["customer"],
            "garment_image_url": test_images["garment"],
            "product_info": product_info
        }
        
        response = requests.post(f"{api_base_url}/tryon/sync", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["completed", "failed"]

    def test_async_processing(self, api_base_url, test_images, product_info):
        """Test asynchronous processing workflow"""
        payload = {
            "api_key": "test-key-12345",
            "customer_image_url": test_images["customer"],
            "garment_image_url": test_images["garment"],
            "product_info": product_info,
            "webhook_url": "https://example.com/webhook"
        }
        
        # Start async processing
        response = requests.post(f"{api_base_url}/tryon/process", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "processing"
        
        # Check status
        max_attempts = 30
        for _ in range(max_attempts):
            status_response = requests.get(f"{api_base_url}/tryon/status/{job_id}")
            status_data = status_response.json()
            
            if status_data["status"] in ["completed", "failed"]:
                break
            time.sleep(2)
        
        assert status_data["status"] in ["completed", "failed"]
        
        # Get result if completed
        if status_data["status"] == "completed":
            result_response = requests.get(f"{api_base_url}/tryon/result/{job_id}")
            assert result_response.status_code == 200

    def test_webhook_functionality(self, api_base_url):
        """Test webhook system"""
        webhook_config = {
            "url": "https://httpbin.org/post",
            "secret": "test-secret"
        }
        
        response = requests.post(f"{api_base_url}/webhook/test", json=webhook_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True

    def test_redis_caching(self):
        """Test Redis caching functionality"""
        test_key = "test:cache:key"
        test_value = {"test": "data", "timestamp": time.time()}
        
        # Set cache
        result = redis_client.set_cache(test_key, test_value, 60)
        assert result is not False
        
        # Get cache
        cached_value = redis_client.get_cache(test_key)
        assert cached_value == test_value
        
        # Delete cache
        redis_client.delete_cache(test_key)
        assert redis_client.get_cache(test_key) is None

    def test_celery_task_processing(self, test_images, product_info):
        """Test Celery task execution"""
        request_data = {
            "api_key": "test-key-12345",
            "customer_image_url": test_images["customer"],
            "garment_image_url": test_images["garment"],
            "product_info": product_info
        }
        
        # Mock the actual processing to avoid external API calls
        with patch('src.integrations.tryon_orchestrator.VirtualTryOnOrchestrator.process_virtual_tryon') as mock_process:
            mock_process.return_value = {
                "success": True,
                "result_image_base64": "data:image/jpeg;base64,test",
                "processing_time": 2.5
            }
            
            # Start task
            task = process_tryon_async.delay(request_data)
            
            # Wait for completion
            result = task.get(timeout=30)
            assert result["success"] is True

    def test_batch_processing(self, api_base_url, test_images, product_info):
        """Test batch processing capability"""
        requests_data = [
            {
                "api_key": "test-key-12345",
                "customer_image_url": test_images["customer"],
                "garment_image_url": test_images["garment"],
                "product_info": product_info
            }
        ] * 3  # 3 identical requests
        
        response = requests.post(f"{api_base_url}/batch-tryon", json=requests_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "task_ids" in data
        assert len(data["task_ids"]) == 3

    def test_error_handling(self, api_base_url):
        """Test error handling and validation"""
        # Test invalid API key
        payload = {
            "api_key": "invalid",
            "customer_image_url": "invalid-url",
            "garment_image_url": "invalid-url",
            "product_info": {}
        }
        
        response = requests.post(f"{api_base_url}/tryon/sync", json=payload)
        assert response.status_code in [400, 401, 500]

    def test_job_cancellation(self, api_base_url, test_images, product_info):
        """Test job cancellation functionality"""
        payload = {
            "api_key": "test-key-12345",
            "customer_image_url": test_images["customer"],
            "garment_image_url": test_images["garment"],
            "product_info": product_info
        }
        
        # Start job
        response = requests.post(f"{api_base_url}/tryon/process", json=payload)
        job_id = response.json()["job_id"]
        
        # Cancel job
        cancel_response = requests.delete(f"{api_base_url}/tryon/job/{job_id}")
        assert cancel_response.status_code == 200

class TestSDKIntegration:
    
    def test_javascript_sdk(self):
        """Test JavaScript SDK functionality"""
        # This would require a browser environment or Node.js
        # For now, just check file exists and has basic structure
        sdk_path = os.path.join(os.path.dirname(__file__), '..', 'sdks', 'javascript', 'virtualfit-sdk.js')
        assert os.path.exists(sdk_path)
        
        with open(sdk_path, 'r') as f:
            content = f.read()
            assert 'class VirtualFitSDK' in content
            assert 'processAsync' in content
            assert 'processSync' in content

    def test_python_sdk(self):
        """Test Python SDK functionality"""
        sdk_path = os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python')
        sys.path.append(sdk_path)
        
        from virtualfit_sdk import VirtualFitSDK
        
        sdk = VirtualFitSDK(api_key="test-key", base_url="http://localhost:8000/api/v1")
        assert sdk.api_key == "test-key"
        assert sdk.base_url == "http://localhost:8000/api/v1"

class TestPluginIntegration:
    
    def test_shopify_plugin_structure(self):
        """Test Shopify plugin file structure"""
        plugin_path = os.path.join(os.path.dirname(__file__), '..', 'plugins', 'shopify', 'virtualfit-app.js')
        assert os.path.exists(plugin_path)
        
        with open(plugin_path, 'r') as f:
            content = f.read()
            assert 'ShopifyVirtualFit' in content
            assert 'addTryOnButtons' in content

    def test_woocommerce_plugin_structure(self):
        """Test WooCommerce plugin file structure"""
        plugin_path = os.path.join(os.path.dirname(__file__), '..', 'plugins', 'woocommerce', 'virtualfit-plugin.php')
        assert os.path.exists(plugin_path)
        
        with open(plugin_path, 'r') as f:
            content = f.read()
            assert 'VirtualFitWooCommerce' in content
            assert 'add_tryon_button' in content

if __name__ == "__main__":
    pytest.main([__file__, "-v"])