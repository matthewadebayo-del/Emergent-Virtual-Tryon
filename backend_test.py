#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Virtual Try-On Application
Tests all endpoints including authentication, products, and virtual try-on functionality
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

class VirtualTryOnAPITester:
    def __init__(self, base_url="https://virtufit-7.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        
        # Test data
        self.test_email = f"test_user_{int(time.time())}@example.com"
        self.test_password = "TestPassword123!"
        self.test_full_name = "Test User"

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
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
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

    def test_health_check(self):
        """Test API health check"""
        success, response_data, status_code = self.make_request('GET', '/')
        
        if success and response_data.get('status') == 'healthy':
            self.log_test("Health Check", True, f"API is healthy - Status: {status_code}")
        else:
            self.log_test("Health Check", False, f"API health check failed - Status: {status_code}", response_data)

    def test_user_registration(self):
        """Test user registration"""
        registration_data = {
            "email": self.test_email,
            "password": self.test_password,
            "full_name": self.test_full_name
        }
        
        success, response_data, status_code = self.make_request('POST', '/register', registration_data)
        
        if success and response_data.get('access_token'):
            self.token = response_data['access_token']
            self.user_data = response_data.get('user')
            self.log_test("User Registration", True, f"User registered successfully - Status: {status_code}")
            return True
        else:
            self.log_test("User Registration", False, f"Registration failed - Status: {status_code}", response_data)
            return False

    def test_user_login(self):
        """Test user login"""
        login_data = {
            "email": self.test_email,
            "password": self.test_password
        }
        
        success, response_data, status_code = self.make_request('POST', '/login', login_data)
        
        if success and response_data.get('access_token'):
            self.token = response_data['access_token']
            self.user_data = response_data.get('user')
            self.log_test("User Login", True, f"Login successful - Status: {status_code}")
            return True
        else:
            self.log_test("User Login", False, f"Login failed - Status: {status_code}", response_data)
            return False

    def test_password_reset(self):
        """Test password reset functionality"""
        reset_data = {
            "email": self.test_email
        }
        
        success, response_data, status_code = self.make_request('POST', '/reset-password', reset_data)
        
        if success and response_data.get('success'):
            self.log_test("Password Reset", True, f"Password reset request successful - Status: {status_code}")
        else:
            self.log_test("Password Reset", False, f"Password reset failed - Status: {status_code}", response_data)

    def test_get_profile(self):
        """Test get user profile"""
        if not self.token:
            self.log_test("Get Profile", False, "No authentication token available")
            return
        
        success, response_data, status_code = self.make_request('GET', '/profile')
        
        if success and response_data.get('email') == self.test_email:
            self.log_test("Get Profile", True, f"Profile retrieved successfully - Status: {status_code}")
        else:
            self.log_test("Get Profile", False, f"Profile retrieval failed - Status: {status_code}", response_data)

    def test_extract_measurements(self):
        """Test measurement extraction from photo"""
        if not self.token:
            self.log_test("Extract Measurements", False, "No authentication token available")
            return
        
        # Create a dummy image file
        dummy_image = b"dummy_image_data_for_testing"
        files = {'user_photo': ('test_photo.jpg', io.BytesIO(dummy_image), 'image/jpeg')}
        
        success, response_data, status_code = self.make_request('POST', '/extract-measurements', 
                                                               files=files, expected_status=200)
        
        if success and response_data.get('success'):
            measurements = response_data.get('data', {})
            if measurements.get('height') and measurements.get('chest'):
                self.log_test("Extract Measurements", True, f"Measurements extracted - Status: {status_code}")
            else:
                self.log_test("Extract Measurements", False, "No measurement data returned", response_data)
        else:
            self.log_test("Extract Measurements", False, f"Measurement extraction failed - Status: {status_code}", response_data)

    def test_get_products(self):
        """Test product catalog retrieval"""
        success, response_data, status_code = self.make_request('GET', '/products')
        
        if success and 'products' in response_data:
            products = response_data['products']
            total = response_data.get('total', 0)
            self.log_test("Get Products", True, f"Retrieved {len(products)} products (total: {total}) - Status: {status_code}")
            return products
        else:
            self.log_test("Get Products", False, f"Product retrieval failed - Status: {status_code}", response_data)
            return []

    def test_get_product_by_id(self, product_id: str):
        """Test getting specific product by ID"""
        success, response_data, status_code = self.make_request('GET', f'/products/{product_id}')
        
        if success and response_data.get('id') == product_id:
            self.log_test("Get Product by ID", True, f"Product retrieved successfully - Status: {status_code}")
            return response_data
        else:
            self.log_test("Get Product by ID", False, f"Product retrieval failed - Status: {status_code}", response_data)
            return None

    def test_virtual_tryon_hybrid(self, product_id: str):
        """Test virtual try-on with hybrid service"""
        if not self.token:
            self.log_test("Virtual Try-On (Hybrid)", False, "No authentication token available")
            return
        
        # Create realistic person image data (base64 encoded sample image)
        realistic_person_image = self._create_realistic_person_image()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Blue'
        }
        
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(realistic_person_image), 'image/jpeg')}
        
        # Measure processing time
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            if result_data.get('result_image_url') and result_data.get('service_type') == 'hybrid':
                cost = result_data.get('cost', 0)
                actual_processing_time = result_data.get('processing_time', processing_time)
                
                # Verify production 3D pipeline criteria
                details = f"Cost: ${cost}, Processing Time: {actual_processing_time:.1f}s, Status: {status_code}"
                
                # Check if it meets production 3D criteria
                if cost >= 0.03 and actual_processing_time >= 15:
                    self.log_test("Virtual Try-On (Hybrid 3D Production)", True, details)
                else:
                    self.log_test("Virtual Try-On (Hybrid 3D Production)", False, 
                                f"Does not meet 3D criteria - {details}")
                
                return result_data
            else:
                self.log_test("Virtual Try-On (Hybrid)", False, "Invalid try-on result data", response_data)
        else:
            self.log_test("Virtual Try-On (Hybrid)", False, f"Try-on failed - Status: {status_code}", response_data)
        
        return None

    def test_virtual_tryon_premium(self, product_id: str):
        """Test virtual try-on with premium fal.ai service"""
        if not self.token:
            self.log_test("Virtual Try-On (Premium)", False, "No authentication token available")
            return
        
        # Create dummy image data
        dummy_image = b"dummy_user_photo_for_premium_tryon_testing"
        
        form_data = {
            'product_id': product_id,
            'service_type': 'premium',
            'size': 'L',
            'color': 'Black'
        }
        
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(dummy_image), 'image/jpeg')}
        
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            if result_data.get('result_image_url'):
                self.log_test("Virtual Try-On (Premium)", True, 
                            f"Premium try-on completed - Cost: ${result_data.get('cost', 0)} - Status: {status_code}")
                return result_data
            else:
                self.log_test("Virtual Try-On (Premium)", False, "Invalid premium try-on result data", response_data)
        else:
            self.log_test("Virtual Try-On (Premium)", False, f"Premium try-on failed - Status: {status_code}", response_data)
        
        return None

    def test_tryon_history(self):
        """Test try-on history retrieval"""
        if not self.token:
            self.log_test("Try-On History", False, "No authentication token available")
            return
        
        success, response_data, status_code = self.make_request('GET', '/tryon-history')
        
        if success and response_data.get('success'):
            history = response_data.get('data', [])
            self.log_test("Try-On History", True, f"Retrieved {len(history)} history items - Status: {status_code}")
        else:
            self.log_test("Try-On History", False, f"History retrieval failed - Status: {status_code}", response_data)

    def _create_realistic_person_image(self) -> bytes:
        """Create a realistic person image for testing"""
        # Create a simple but realistic-looking person silhouette
        # This simulates a real person photo for testing the 3D pipeline
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Create a 512x512 image with a person-like shape
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a person-like silhouette
        # Head (circle)
        draw.ellipse([200, 50, 312, 162], fill='#D2B48C')  # Skin tone
        
        # Body (rectangle with rounded edges)
        draw.rectangle([180, 160, 332, 400], fill='#4169E1')  # Blue shirt
        
        # Arms
        draw.rectangle([120, 180, 180, 350], fill='#4169E1')  # Left arm
        draw.rectangle([332, 180, 392, 350], fill='#4169E1')  # Right arm
        
        # Legs
        draw.rectangle([190, 400, 240, 500], fill='#2F4F4F')  # Left leg (dark pants)
        draw.rectangle([272, 400, 322, 500], fill='#2F4F4F')  # Right leg
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return buffer.getvalue()

    def test_hybrid_3d_pipeline_comprehensive(self):
        """Comprehensive test of the Hybrid 3D Pipeline"""
        print("\n🔬 Testing Production-Ready Hybrid 3D Virtual Try-On Pipeline")
        print("=" * 70)
        
        if not self.token:
            self.log_test("Hybrid 3D Pipeline", False, "No authentication token available")
            return
        
        # Get products for testing different categories
        products = self.test_get_products()
        if not products:
            self.log_test("Hybrid 3D Pipeline", False, "No products available for testing")
            return
        
        # Test different product categories
        categories_to_test = ['tops', 'bottoms', 'dresses', 'outerwear']
        tested_categories = []
        
        for product in products[:6]:  # Test first 6 products
            category = product.get('category', '').lower()
            product_id = product.get('id')
            product_name = product.get('name', 'Unknown Product')
            
            if not product_id:
                continue
                
            print(f"\n🧪 Testing {product_name} ({category})")
            
            # Create realistic person image
            realistic_person_image = self._create_realistic_person_image()
            
            form_data = {
                'product_id': product_id,
                'service_type': 'hybrid',  # Force hybrid 3D processing
                'size': 'M',
                'color': 'Default'
            }
            
            files = {'user_photo': ('realistic_person.jpg', io.BytesIO(realistic_person_image), 'image/jpeg')}
            
            # Measure processing time
            start_time = time.time()
            success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                                   data=form_data, files=files, expected_status=200)
            processing_time = time.time() - start_time
            
            if success and response_data.get('success'):
                result_data = response_data.get('data', {})
                cost = result_data.get('cost', 0)
                actual_processing_time = result_data.get('processing_time', processing_time)
                result_url = result_data.get('result_image_url', '')
                
                # Verify Hybrid 3D Pipeline criteria
                criteria_met = []
                criteria_failed = []
                
                # 1. Check cost reflects production 3D pipeline ($0.03)
                if cost >= 0.03:
                    criteria_met.append(f"✅ Production cost: ${cost}")
                else:
                    criteria_failed.append(f"❌ Cost too low: ${cost} (expected ≥$0.03)")
                
                # 2. Check processing time (15-30 seconds for real 3D)
                if 15 <= actual_processing_time <= 35:
                    criteria_met.append(f"✅ Real 3D processing time: {actual_processing_time:.1f}s")
                elif actual_processing_time >= 10:
                    criteria_met.append(f"⚠️ Processing time: {actual_processing_time:.1f}s (acceptable)")
                else:
                    criteria_failed.append(f"❌ Processing too fast: {actual_processing_time:.1f}s (expected 15-30s)")
                
                # 3. Check result is data URL (indicates real processing)
                if result_url.startswith('data:image/'):
                    criteria_met.append("✅ Real image processing (data URL)")
                elif result_url.startswith('http'):
                    criteria_met.append("✅ Image result generated")
                else:
                    criteria_failed.append("❌ Invalid result format")
                
                # 4. Check service type
                if result_data.get('service_type') == 'hybrid':
                    criteria_met.append("✅ Hybrid service confirmed")
                else:
                    criteria_failed.append("❌ Wrong service type")
                
                # Log results
                if len(criteria_failed) == 0:
                    self.log_test(f"Hybrid 3D - {product_name}", True, 
                                f"All criteria met: {'; '.join(criteria_met)}")
                    tested_categories.append(category)
                else:
                    self.log_test(f"Hybrid 3D - {product_name}", False, 
                                f"Failed criteria: {'; '.join(criteria_failed)}")
                
                # Print detailed results
                print(f"   Cost: ${cost}")
                print(f"   Processing Time: {actual_processing_time:.1f}s")
                print(f"   Result Type: {'Data URL' if result_url.startswith('data:') else 'External URL'}")
                
            else:
                self.log_test(f"Hybrid 3D - {product_name}", False, 
                            f"Request failed - Status: {status_code}")
                print(f"   ❌ Request failed: {response_data}")
        
        # Summary of category testing
        print(f"\n📊 Categories tested: {len(set(tested_categories))}")
        print(f"   Tested: {', '.join(set(tested_categories))}")
        
        # Overall assessment
        if len(set(tested_categories)) >= 2:
            self.log_test("Hybrid 3D Pipeline - Category Coverage", True, 
                        f"Successfully tested {len(set(tested_categories))} different categories")
        else:
            self.log_test("Hybrid 3D Pipeline - Category Coverage", False, 
                        "Insufficient category coverage")

    def test_hybrid_3d_vs_premium_comparison(self):
        """Compare Hybrid 3D vs Premium fal.ai processing"""
        print("\n⚖️ Comparing Hybrid 3D vs Premium fal.ai Processing")
        print("=" * 60)
        
        if not self.token:
            self.log_test("3D vs Premium Comparison", False, "No authentication token available")
            return
        
        # Get a product for testing
        products = self.test_get_products()
        if not products:
            self.log_test("3D vs Premium Comparison", False, "No products available")
            return
        
        test_product = products[0]
        product_id = test_product.get('id')
        product_name = test_product.get('name', 'Test Product')
        
        if not product_id:
            self.log_test("3D vs Premium Comparison", False, "No valid product ID")
            return
        
        realistic_person_image = self._create_realistic_person_image()
        
        # Test Hybrid 3D
        print(f"\n🔬 Testing Hybrid 3D with {product_name}")
        hybrid_result = self._test_service_type(product_id, 'hybrid', realistic_person_image)
        
        # Test Premium fal.ai
        print(f"\n💎 Testing Premium fal.ai with {product_name}")
        premium_result = self._test_service_type(product_id, 'premium', realistic_person_image)
        
        # Compare results
        if hybrid_result and premium_result:
            hybrid_cost = hybrid_result.get('cost', 0)
            premium_cost = premium_result.get('cost', 0)
            hybrid_time = hybrid_result.get('processing_time', 0)
            premium_time = premium_result.get('processing_time', 0)
            
            comparison_details = (
                f"Hybrid 3D: ${hybrid_cost}, {hybrid_time:.1f}s | "
                f"Premium: ${premium_cost}, {premium_time:.1f}s"
            )
            
            # Verify expected cost differences
            if hybrid_cost >= 0.03 and premium_cost >= 0.07:
                self.log_test("Service Comparison", True, 
                            f"Cost structure correct - {comparison_details}")
            else:
                self.log_test("Service Comparison", False, 
                            f"Unexpected cost structure - {comparison_details}")
        else:
            self.log_test("Service Comparison", False, "Could not complete comparison")

    def _test_service_type(self, product_id: str, service_type: str, image_data: bytes) -> dict:
        """Test a specific service type and return results"""
        form_data = {
            'product_id': product_id,
            'service_type': service_type,
            'size': 'M',
            'color': 'Default'
        }
        
        files = {'user_photo': (f'{service_type}_test.jpg', io.BytesIO(image_data), 'image/jpeg')}
        
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            result_data['actual_processing_time'] = processing_time
            
            cost = result_data.get('cost', 0)
            proc_time = result_data.get('processing_time', processing_time)
            
            print(f"   Result: ${cost}, {proc_time:.1f}s processing")
            return result_data
        else:
            print(f"   Failed: {response_data}")
            return None

    def test_invalid_authentication(self):
        """Test API behavior with invalid authentication"""
        # Save current token
        original_token = self.token
        
        # Test with invalid token
        self.token = "invalid_token_12345"
        success, response_data, status_code = self.make_request('GET', '/profile', expected_status=401)
        
        if status_code == 401:
            self.log_test("Invalid Authentication", True, "API correctly rejected invalid token")
        else:
            self.log_test("Invalid Authentication", False, f"API should reject invalid token - Status: {status_code}")
        
        # Restore original token
        self.token = original_token

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting Virtual Try-On Backend API Tests")
        print(f"📍 Testing API at: {self.base_url}")
        print("=" * 60)
        
        # Basic connectivity tests
        self.test_health_check()
        
        # Authentication tests
        if self.test_user_registration():
            self.test_get_profile()
            self.test_extract_measurements()
        
        # Test login separately (in case registration fails)
        self.test_user_login()
        self.test_password_reset()
        
        # Product catalog tests
        products = self.test_get_products()
        if products:
            # Test getting specific product
            first_product = products[0]
            product_id = first_product.get('id')
            if product_id:
                product_details = self.test_get_product_by_id(product_id)
                
                # Virtual try-on tests
                if product_details:
                    self.test_virtual_tryon_hybrid(product_id)
                    self.test_virtual_tryon_premium(product_id)
        
        # PRODUCTION-READY HYBRID 3D PIPELINE TESTS
        print("\n" + "="*70)
        print("🔬 PRODUCTION-READY HYBRID 3D PIPELINE TESTING")
        print("="*70)
        
        # Comprehensive Hybrid 3D Pipeline test
        self.test_hybrid_3d_pipeline_comprehensive()
        
        # Service comparison test
        self.test_hybrid_3d_vs_premium_comparison()
        
        # Try-on history test
        self.test_tryon_history()
        
        # Security tests
        self.test_invalid_authentication()
        
        # Print final results
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("\n🎉 ALL TESTS PASSED! Backend API is working correctly.")
        else:
            print(f"\n⚠️  {self.tests_run - self.tests_passed} tests failed. Check the details above.")
        
        # Print failed tests
        failed_tests = [test for test in self.test_results if not test['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        print("\n" + "=" * 60)

def main():
    """Main test execution"""
    tester = VirtualTryOnAPITester()
    tester.run_all_tests()
    
    # Return appropriate exit code
    if tester.tests_passed == tester.tests_run:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())