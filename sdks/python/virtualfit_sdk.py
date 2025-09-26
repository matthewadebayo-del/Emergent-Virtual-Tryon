"""
VirtualFit Python SDK
E-commerce integration for virtual try-on
"""
import requests
import json
from typing import Dict, Optional, Any

class VirtualFitSDK:
    def __init__(self, api_key: str, base_url: str = "https://api.virtualfit.com/api/v1", webhook_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.webhook_url = webhook_url
        self.session = requests.Session()
    
    def process_async(self, customer_image_url: str, garment_image_url: str, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process virtual try-on asynchronously"""
        payload = {
            "api_key": self.api_key,
            "customer_image_url": customer_image_url,
            "garment_image_url": garment_image_url,
            "product_info": product_info,
            "webhook_url": self.webhook_url
        }
        
        response = self.session.post(f"{self.base_url}/tryon/process", json=payload)
        response.raise_for_status()
        return response.json()
    
    def process_sync(self, customer_image_url: str, garment_image_url: str, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process virtual try-on synchronously"""
        payload = {
            "api_key": self.api_key,
            "customer_image_url": customer_image_url,
            "garment_image_url": garment_image_url,
            "product_info": product_info
        }
        
        response = self.session.post(f"{self.base_url}/tryon/sync", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get processing status"""
        response = self.session.get(f"{self.base_url}/tryon/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Get processing result"""
        response = self.session.get(f"{self.base_url}/tryon/result/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def batch_process(self, requests_data: list) -> Dict[str, Any]:
        """Process multiple try-on requests"""
        response = self.session.post(f"{self.base_url}/batch-tryon", json=requests_data)
        response.raise_for_status()
        return response.json()

# Django integration helper
class DjangoVirtualFitMixin:
    """Mixin for Django models"""
    
    def get_tryon_result(self, customer_image_field: str, garment_image_field: str):
        sdk = VirtualFitSDK(api_key=settings.VIRTUALFIT_API_KEY)
        
        customer_url = getattr(self, customer_image_field).url
        garment_url = getattr(self, garment_image_field).url
        
        product_info = {
            "name": getattr(self, 'name', 'Product'),
            "category": getattr(self, 'category', 'TOP'),
            "color": getattr(self, 'color', 'white')
        }
        
        return sdk.process_sync(customer_url, garment_url, product_info)

# Flask integration helper  
def create_flask_blueprint(sdk: VirtualFitSDK):
    """Create Flask blueprint for VirtualFit integration"""
    from flask import Blueprint, request, jsonify
    
    bp = Blueprint('virtualfit', __name__)
    
    @bp.route('/tryon', methods=['POST'])
    def process_tryon():
        data = request.json
        result = sdk.process_async(
            data['customer_image_url'],
            data['garment_image_url'], 
            data['product_info']
        )
        return jsonify(result)
    
    @bp.route('/tryon/status/<job_id>')
    def get_status(job_id):
        result = sdk.get_status(job_id)
        return jsonify(result)
    
    return bp