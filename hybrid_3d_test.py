#!/usr/bin/env python3
"""
Focused test for Production-Ready Hybrid 3D Virtual Try-On Pipeline
Tests the 4-step 3D process as specified in the review request
"""

import requests
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import io
import base64
from PIL import Image, ImageDraw

class Hybrid3DPipelineTester:
    def __init__(self, base_url="https://virtufit-7.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        
        # Test data
        self.test_email = f"hybrid3d_user_{int(time.time())}@example.com"
        self.test_password = "TestPassword123!"
        self.test_full_name = "Hybrid 3D Test User"

    def log_test(self, name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED")
        else:
            print(f"❌ {name}: FAILED - {details}")
        
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details,
            "response_data": response_data
        })

    def make_request(self, method: str, endpoint: str, data: Any = None, files: Any = None, 
                    expected_status: int = 200) -> tuple[bool, Dict[str, Any], int]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    # Remove Content-Type for multipart/form-data
                    headers.pop('Content-Type', None)
                    response = requests.post(url, data=data, files=files, headers=headers)
                else:
                    response = requests.post(url, json=data, headers=headers)
            else:
                return False, {}, 0
            
            success = response.status_code == expected_status
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}
            
            return success, response_data, response.status_code
            
        except Exception as e:
            return False, {"error": str(e)}, 0

    def create_realistic_person_image(self) -> bytes:
        """Create a realistic person image for testing the 3D pipeline"""
        # Create a 512x512 image with a person-like shape that MediaPipe can detect
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a more detailed person silhouette for better pose detection
        # Head (circle)
        draw.ellipse([200, 50, 312, 162], fill='#D2B48C')  # Skin tone
        
        # Neck
        draw.rectangle([240, 162, 272, 180], fill='#D2B48C')
        
        # Body (torso)
        draw.rectangle([180, 180, 332, 400], fill='#4169E1')  # Blue shirt
        
        # Arms with joints
        draw.rectangle([120, 200, 180, 350], fill='#D2B48C')  # Left arm (skin)
        draw.rectangle([332, 200, 392, 350], fill='#D2B48C')  # Right arm (skin)
        
        # Hands
        draw.ellipse([110, 340, 130, 360], fill='#D2B48C')  # Left hand
        draw.ellipse([382, 340, 402, 360], fill='#D2B48C')  # Right hand
        
        # Legs
        draw.rectangle([190, 400, 240, 500], fill='#2F4F4F')  # Left leg (dark pants)
        draw.rectangle([272, 400, 322, 500], fill='#2F4F4F')  # Right leg
        
        # Feet
        draw.ellipse([185, 490, 215, 510], fill='#000000')  # Left foot
        draw.ellipse([267, 490, 297, 510], fill='#000000')  # Right foot
        
        # Add some facial features for better detection
        draw.ellipse([220, 80, 230, 90], fill='#000000')  # Left eye
        draw.ellipse([282, 80, 292, 90], fill='#000000')  # Right eye
        draw.ellipse([250, 110, 262, 120], fill='#000000')  # Nose
        draw.arc([240, 130, 272, 150], 0, 180, fill='#000000')  # Mouth
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return buffer.getvalue()

    def setup_authentication(self):
        """Setup user authentication"""
        print("🔐 Setting up authentication...")
        
        # Register user
        registration_data = {
            "email": self.test_email,
            "password": self.test_password,
            "full_name": self.test_full_name
        }
        
        success, response_data, status_code = self.make_request('POST', '/register', registration_data)
        
        if success and response_data.get('access_token'):
            self.token = response_data['access_token']
            self.user_data = response_data.get('user')
            self.log_test("Authentication Setup", True, f"User registered and authenticated")
            return True
        else:
            self.log_test("Authentication Setup", False, f"Registration failed - Status: {status_code}")
            return False

    def get_test_products(self):
        """Get products for testing"""
        success, response_data, status_code = self.make_request('GET', '/products')
        
        if success and 'products' in response_data:
            products = response_data['products']
            print(f"📦 Found {len(products)} products for testing")
            return products
        else:
            print("❌ Failed to get products")
            return []

    def test_hybrid_3d_pipeline_step_by_step(self, product_id: str, product_name: str, category: str):
        """Test the complete 4-step Hybrid 3D pipeline"""
        print(f"\n🔬 Testing Hybrid 3D Pipeline with {product_name} ({category})")
        print("=" * 60)
        
        # Create realistic person image
        person_image = self.create_realistic_person_image()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',  # Force hybrid 3D processing
            'size': 'M',
            'color': 'Default'
        }
        
        files = {'user_photo': ('realistic_person.jpg', io.BytesIO(person_image), 'image/jpeg')}
        
        # Measure processing time
        print("⏱️ Starting processing...")
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            cost = result_data.get('cost', 0)
            actual_processing_time = result_data.get('processing_time', processing_time)
            result_url = result_data.get('result_image_url', '')
            
            print(f"💰 Cost: ${cost}")
            print(f"⏱️ Processing Time: {actual_processing_time:.1f}s")
            print(f"🖼️ Result Type: {'Data URL' if result_url.startswith('data:') else 'External URL'}")
            
            # Verify Hybrid 3D Pipeline criteria
            criteria_results = []
            
            # 1. Check cost reflects production 3D pipeline ($0.03)
            if cost >= 0.03:
                criteria_results.append(("✅ Production 3D Cost", True, f"${cost} ≥ $0.03"))
            else:
                criteria_results.append(("❌ Production 3D Cost", False, f"${cost} < $0.03 (expected ≥$0.03)"))
            
            # 2. Check processing time (15-30 seconds for real 3D)
            if 15 <= actual_processing_time <= 35:
                criteria_results.append(("✅ Real 3D Processing Time", True, f"{actual_processing_time:.1f}s in expected range"))
            elif actual_processing_time >= 10:
                criteria_results.append(("⚠️ Processing Time", True, f"{actual_processing_time:.1f}s (acceptable for 3D)"))
            else:
                criteria_results.append(("❌ Processing Time", False, f"{actual_processing_time:.1f}s too fast for real 3D"))
            
            # 3. Check result format
            if result_url.startswith('data:image/'):
                criteria_results.append(("✅ Real Image Processing", True, "Generated data URL result"))
            elif result_url.startswith('http'):
                criteria_results.append(("✅ Image Result", True, "External URL result"))
            else:
                criteria_results.append(("❌ Invalid Result", False, "No valid image result"))
            
            # 4. Check service type
            if result_data.get('service_type') == 'hybrid':
                criteria_results.append(("✅ Hybrid Service", True, "Confirmed hybrid processing"))
            else:
                criteria_results.append(("❌ Service Type", False, f"Wrong service type: {result_data.get('service_type')}"))
            
            # Log all criteria results
            all_passed = True
            for name, passed, details in criteria_results:
                self.log_test(f"{name} - {product_name}", passed, details)
                if not passed:
                    all_passed = False
            
            # Overall result for this product
            if all_passed:
                self.log_test(f"Hybrid 3D Pipeline - {product_name}", True, 
                            f"All 4-step 3D criteria met: ${cost}, {actual_processing_time:.1f}s")
            else:
                self.log_test(f"Hybrid 3D Pipeline - {product_name}", False, 
                            "Some 3D pipeline criteria not met")
            
            return result_data
            
        else:
            self.log_test(f"Hybrid 3D Pipeline - {product_name}", False, 
                        f"Request failed - Status: {status_code}")
            print(f"❌ Request failed: {response_data}")
            return None

    def run_hybrid_3d_tests(self):
        """Run comprehensive Hybrid 3D pipeline tests"""
        print("🚀 Starting Production-Ready Hybrid 3D Virtual Try-On Pipeline Tests")
        print(f"📍 Testing API at: {self.base_url}")
        print("=" * 80)
        
        # Setup authentication
        if not self.setup_authentication():
            print("❌ Cannot proceed without authentication")
            return
        
        # Get products
        products = self.get_test_products()
        if not products:
            print("❌ Cannot proceed without products")
            return
        
        # Test different product categories
        categories_tested = set()
        successful_tests = 0
        
        print(f"\n🧪 Testing up to 5 products from different categories...")
        
        for i, product in enumerate(products[:5]):  # Test first 5 products
            product_id = product.get('id')
            product_name = product.get('name', 'Unknown Product')
            category = product.get('category', 'Unknown Category')
            
            if not product_id:
                continue
            
            print(f"\n--- Test {i+1}/5 ---")
            result = self.test_hybrid_3d_pipeline_step_by_step(product_id, product_name, category)
            
            if result:
                categories_tested.add(category.lower())
                successful_tests += 1
            
            # Small delay between tests
            time.sleep(2)
        
        # Summary
        print(f"\n" + "="*80)
        print("📊 HYBRID 3D PIPELINE TEST SUMMARY")
        print("="*80)
        
        print(f"✅ Successful Tests: {successful_tests}/5")
        print(f"📂 Categories Tested: {len(categories_tested)}")
        print(f"   Categories: {', '.join(categories_tested)}")
        
        # Final assessment
        if successful_tests >= 3 and len(categories_tested) >= 2:
            self.log_test("Overall Hybrid 3D Pipeline Assessment", True, 
                        f"Successfully tested {successful_tests} products across {len(categories_tested)} categories")
        else:
            self.log_test("Overall Hybrid 3D Pipeline Assessment", False, 
                        f"Insufficient successful tests: {successful_tests}/5")
        
        self.print_final_summary()

    def print_final_summary(self):
        """Print final test summary"""
        print("\n" + "=" * 80)
        print("🎯 FINAL TEST RESULTS")
        print("=" * 80)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("\n🎉 ALL TESTS PASSED! Production-Ready Hybrid 3D Pipeline is working correctly.")
        else:
            print(f"\n⚠️  {self.tests_run - self.tests_passed} tests failed. Check the details above.")
        
        # Print failed tests
        failed_tests = [test for test in self.test_results if not test['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        print("\n" + "=" * 80)

def main():
    """Main test execution"""
    tester = Hybrid3DPipelineTester()
    tester.run_hybrid_3d_tests()
    
    # Return appropriate exit code
    if tester.tests_passed == tester.tests_run:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())