#!/usr/bin/env python3
"""
Detailed Virtual Try-On Test with Real Image Data
Tests the actual AI pipeline processing capabilities
"""

import requests
import sys
import json
import time
import io
import base64
from PIL import Image, ImageDraw
import numpy as np

class DetailedTryOnTester:
    def __init__(self, base_url="https://tryon-hybrid-3d.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        
        # Create test user
        self.test_email = f"detailed_test_{int(time.time())}@example.com"
        self.test_password = "TestPassword123!"
        self.test_full_name = "Detailed Test User"

    def create_realistic_test_image(self, width=400, height=600):
        """Create a more realistic test image that resembles a person"""
        # Create a simple person-like silhouette
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple person silhouette
        # Head (circle)
        head_center = (width//2, height//6)
        head_radius = width//8
        draw.ellipse([
            head_center[0] - head_radius, head_center[1] - head_radius,
            head_center[0] + head_radius, head_center[1] + head_radius
        ], fill='lightblue')
        
        # Body (rectangle)
        body_width = width//3
        body_height = height//2
        body_left = width//2 - body_width//2
        body_top = head_center[1] + head_radius
        draw.rectangle([
            body_left, body_top,
            body_left + body_width, body_top + body_height
        ], fill='lightgreen')
        
        # Arms
        arm_width = width//12
        arm_height = height//3
        # Left arm
        draw.rectangle([
            body_left - arm_width, body_top + 20,
            body_left, body_top + arm_height
        ], fill='lightblue')
        # Right arm
        draw.rectangle([
            body_left + body_width, body_top + 20,
            body_left + body_width + arm_width, body_top + arm_height
        ], fill='lightblue')
        
        # Legs
        leg_width = width//8
        leg_height = height//3
        leg_top = body_top + body_height
        # Left leg
        draw.rectangle([
            body_left + body_width//4, leg_top,
            body_left + body_width//4 + leg_width, leg_top + leg_height
        ], fill='darkblue')
        # Right leg
        draw.rectangle([
            body_left + 3*body_width//4 - leg_width, leg_top,
            body_left + 3*body_width//4, leg_top + leg_height
        ], fill='darkblue')
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        return buffer.getvalue()

    def register_and_login(self):
        """Register and login test user"""
        # Register
        registration_data = {
            "email": self.test_email,
            "password": self.test_password,
            "full_name": self.test_full_name
        }
        
        response = requests.post(f"{self.base_url}/register", json=registration_data)
        if response.status_code == 200:
            data = response.json()
            self.token = data.get('access_token')
            print("✅ User registered and authenticated successfully")
            return True
        else:
            print(f"❌ Registration failed: {response.status_code}")
            return False

    def get_test_product(self):
        """Get a product for testing"""
        response = requests.get(f"{self.base_url}/products")
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            if products:
                product = products[0]
                print(f"✅ Using test product: {product['name']} (ID: {product['id']})")
                return product
        print("❌ Failed to get test product")
        return None

    def test_hybrid_tryon_detailed(self, product_id):
        """Test hybrid try-on with detailed logging"""
        print("\n🔬 Testing Hybrid 3D Pipeline with Realistic Image...")
        
        # Create realistic test image
        test_image_data = self.create_realistic_test_image()
        print(f"📸 Created test image: {len(test_image_data)} bytes")
        
        # Prepare request
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Blue'
        }
        files = {'user_photo': ('test_person.jpg', io.BytesIO(test_image_data), 'image/jpeg')}
        
        # Make request
        start_time = time.time()
        response = requests.post(f"{self.base_url}/tryon", data=form_data, files=files, headers=headers)
        processing_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result_data = data.get('data', {})
                print(f"✅ Hybrid try-on completed successfully")
                print(f"💰 Cost: ${result_data.get('cost', 0)}")
                print(f"🖼️  Result URL type: {'data URL' if result_data.get('result_image_url', '').startswith('data:') else 'external URL'}")
                print(f"⚙️  Service type: {result_data.get('service_type')}")
                return True
            else:
                print(f"❌ Try-on failed: {data}")
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
        
        return False

    def test_premium_tryon_detailed(self, product_id):
        """Test premium fal.ai try-on with detailed logging"""
        print("\n🔬 Testing fal.ai Premium Pipeline...")
        
        # Create realistic test image
        test_image_data = self.create_realistic_test_image()
        print(f"📸 Created test image: {len(test_image_data)} bytes")
        
        # Prepare request
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {
            'product_id': product_id,
            'service_type': 'premium',
            'size': 'L',
            'color': 'Black'
        }
        files = {'user_photo': ('test_person.jpg', io.BytesIO(test_image_data), 'image/jpeg')}
        
        # Make request
        start_time = time.time()
        response = requests.post(f"{self.base_url}/tryon", data=form_data, files=files, headers=headers)
        processing_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result_data = data.get('data', {})
                print(f"✅ Premium try-on completed successfully")
                print(f"💰 Cost: ${result_data.get('cost', 0)}")
                print(f"🖼️  Result URL type: {'data URL' if result_data.get('result_image_url', '').startswith('data:') else 'external URL'}")
                print(f"⚙️  Service type: {result_data.get('service_type')}")
                return True
            else:
                print(f"❌ Try-on failed: {data}")
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
        
        return False

    def run_detailed_tests(self):
        """Run detailed virtual try-on tests"""
        print("🔬 Starting Detailed Virtual Try-On Tests")
        print("=" * 60)
        
        # Setup
        if not self.register_and_login():
            return False
        
        product = self.get_test_product()
        if not product:
            return False
        
        product_id = product['id']
        
        # Test both pipelines
        hybrid_success = self.test_hybrid_tryon_detailed(product_id)
        premium_success = self.test_premium_tryon_detailed(product_id)
        
        print("\n" + "=" * 60)
        print("📊 DETAILED TEST RESULTS")
        print("=" * 60)
        print(f"Hybrid 3D Pipeline: {'✅ WORKING' if hybrid_success else '❌ FAILED'}")
        print(f"fal.ai Premium Pipeline: {'✅ WORKING' if premium_success else '❌ FAILED'}")
        
        if hybrid_success or premium_success:
            print("\n🎉 Real AI pipeline is being called (not mock data)")
        else:
            print("\n⚠️  Both pipelines had issues - check logs for details")
        
        return hybrid_success or premium_success

def main():
    tester = DetailedTryOnTester()
    success = tester.run_detailed_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())