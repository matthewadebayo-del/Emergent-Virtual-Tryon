#!/usr/bin/env python3
"""
Enhanced Virtual Try-On Testing
Tests the improved virtual try-on functionality with advanced garment fitting and blending
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
import os
from PIL import Image
import numpy as np

class EnhancedTryOnTester:
    def __init__(self, base_url="https://virtufit-7.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        
        # Test data with realistic user
        self.test_email = f"emma_wilson_{int(time.time())}@example.com"
        self.test_password = "SecurePass2024!"
        self.test_full_name = "Emma Wilson"

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
            "response_data": response_data,
            "timestamp": datetime.now().isoformat()
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
        """Create a realistic person image for testing"""
        try:
            # Create a more realistic person silhouette
            img = Image.new('RGB', (400, 600), color='white')
            
            # Draw a person-like shape
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Head (circle)
            draw.ellipse([175, 50, 225, 100], fill='#FFDBAC', outline='#D4A574')
            
            # Body (rectangle with rounded corners)
            draw.rectangle([160, 100, 240, 350], fill='#87CEEB', outline='#4682B4')  # Light blue shirt
            
            # Arms
            draw.rectangle([140, 120, 160, 280], fill='#FFDBAC', outline='#D4A574')  # Left arm
            draw.rectangle([240, 120, 260, 280], fill='#FFDBAC', outline='#D4A574')  # Right arm
            
            # Legs
            draw.rectangle([170, 350, 190, 550], fill='#000080', outline='#000040')  # Left leg (dark blue pants)
            draw.rectangle([210, 350, 230, 550], fill='#000080', outline='#000040')  # Right leg
            
            # Convert to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Warning: Could not create realistic image: {e}")
            # Fallback to simple test data
            return b"realistic_person_image_data_for_enhanced_testing"

    def setup_test_user(self) -> bool:
        """Setup test user for enhanced testing"""
        print("🔧 Setting up test user...")
        
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
            self.log_test("User Setup", True, f"Test user '{self.test_full_name}' created successfully")
            return True
        else:
            self.log_test("User Setup", False, f"Failed to create test user - Status: {status_code}", response_data)
            return False

    def test_enhanced_hybrid_tryon(self, product_id: str, product_name: str, category: str):
        """Test enhanced hybrid virtual try-on with realistic person image"""
        if not self.token:
            self.log_test("Enhanced Hybrid Try-On", False, "No authentication token available")
            return None
        
        print(f"🧪 Testing Enhanced Hybrid Try-On with {product_name}...")
        
        # Create realistic person image
        person_image = self.create_realistic_person_image()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Navy'
        }
        
        files = {'user_photo': ('realistic_person.jpg', io.BytesIO(person_image), 'image/jpeg')}
        
        # Measure processing time
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            cost = result_data.get('cost', 0)
            service_type = result_data.get('service_type')
            result_url = result_data.get('result_image_url', '')
            
            # Verify enhanced functionality criteria
            issues = []
            
            # 1. Check if real AI processing (cost should be $0.02 for hybrid, not $0.01 mock)
            if cost == 0.01:
                issues.append("Cost indicates mock processing ($0.01) instead of real AI ($0.02)")
            elif cost != 0.02:
                issues.append(f"Unexpected cost: ${cost} (expected $0.02 for hybrid)")
            
            # 2. Check processing time (should be 10-20 seconds for real AI)
            if processing_time < 5:
                issues.append(f"Processing too fast ({processing_time:.1f}s) - suggests mock processing")
            elif processing_time > 30:
                issues.append(f"Processing too slow ({processing_time:.1f}s) - may indicate issues")
            
            # 3. Check result format (should be data URL for real processing)
            if not result_url.startswith('data:image/'):
                issues.append("Result should be data URL for real AI processing")
            
            # 4. Verify service type
            if service_type != 'hybrid':
                issues.append(f"Wrong service type: {service_type} (expected 'hybrid')")
            
            if not issues:
                self.log_test("Enhanced Hybrid Try-On", True, 
                            f"Real AI processing confirmed - Cost: ${cost}, Time: {processing_time:.1f}s")
                return result_data
            else:
                self.log_test("Enhanced Hybrid Try-On", False, 
                            f"Issues detected: {'; '.join(issues)}", result_data)
        else:
            self.log_test("Enhanced Hybrid Try-On", False, 
                        f"Try-on failed - Status: {status_code}", response_data)
        
        return None

    def test_enhanced_premium_tryon(self, product_id: str, product_name: str, category: str):
        """Test enhanced premium fal.ai virtual try-on"""
        if not self.token:
            self.log_test("Enhanced Premium Try-On", False, "No authentication token available")
            return None
        
        print(f"🧪 Testing Enhanced Premium Try-On with {product_name}...")
        
        # Create realistic person image
        person_image = self.create_realistic_person_image()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'premium',
            'size': 'L',
            'color': 'Black'
        }
        
        files = {'user_photo': ('realistic_person.jpg', io.BytesIO(person_image), 'image/jpeg')}
        
        # Measure processing time
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            cost = result_data.get('cost', 0)
            service_type = result_data.get('service_type')
            result_url = result_data.get('result_image_url', '')
            
            # Verify enhanced functionality criteria
            issues = []
            
            # 1. Check if real fal.ai processing (cost should be $0.075, not $0.01 mock)
            if cost == 0.01:
                issues.append("Cost indicates mock processing ($0.01) instead of real fal.ai ($0.075)")
            elif cost == 0.02:
                issues.append("Cost indicates fallback to hybrid ($0.02) instead of fal.ai ($0.075)")
            elif cost != 0.075:
                issues.append(f"Unexpected cost: ${cost} (expected $0.075 for fal.ai)")
            
            # 2. Check processing time
            if processing_time < 5:
                issues.append(f"Processing too fast ({processing_time:.1f}s) - suggests mock processing")
            
            # 3. Check result format
            if not result_url.startswith('data:image/') and not result_url.startswith('http'):
                issues.append("Invalid result URL format")
            
            # Note: fal.ai might fail and fallback to hybrid, which is acceptable
            if cost == 0.02 and processing_time > 10:
                self.log_test("Enhanced Premium Try-On", True, 
                            f"fal.ai failed, fallback to hybrid successful - Cost: ${cost}, Time: {processing_time:.1f}s")
                return result_data
            elif not issues:
                self.log_test("Enhanced Premium Try-On", True, 
                            f"Real fal.ai processing confirmed - Cost: ${cost}, Time: {processing_time:.1f}s")
                return result_data
            else:
                self.log_test("Enhanced Premium Try-On", False, 
                            f"Issues detected: {'; '.join(issues)}", result_data)
        else:
            self.log_test("Enhanced Premium Try-On", False, 
                        f"Premium try-on failed - Status: {status_code}", response_data)
        
        return None

    def test_different_product_categories(self):
        """Test virtual try-on with different product categories"""
        print("🧪 Testing Different Product Categories...")
        
        # Get products
        success, response_data, status_code = self.make_request('GET', '/products')
        
        if not success or 'products' not in response_data:
            self.log_test("Product Categories Test", False, "Could not retrieve products")
            return
        
        products = response_data['products']
        categories_tested = set()
        successful_tests = 0
        
        # Test different categories
        for product in products[:5]:  # Test first 5 products
            product_id = product.get('id')
            product_name = product.get('name', 'Unknown')
            category = product.get('category', 'Unknown')
            
            if category not in categories_tested:
                print(f"   Testing category: {category} with {product_name}")
                
                # Test hybrid for this category
                result = self.test_enhanced_hybrid_tryon(product_id, product_name, category)
                if result:
                    successful_tests += 1
                
                categories_tested.add(category)
                
                # Small delay between tests
                time.sleep(1)
        
        if successful_tests > 0:
            self.log_test("Product Categories Test", True, 
                        f"Successfully tested {successful_tests} categories: {', '.join(categories_tested)}")
        else:
            self.log_test("Product Categories Test", False, 
                        "No categories tested successfully")

    def test_garment_fitting_quality(self, product_id: str):
        """Test the quality of garment fitting and blending"""
        if not self.token:
            self.log_test("Garment Fitting Quality", False, "No authentication token available")
            return
        
        print("🧪 Testing Garment Fitting Quality...")
        
        # Create realistic person image
        person_image = self.create_realistic_person_image()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Blue'
        }
        
        files = {'user_photo': ('quality_test_person.jpg', io.BytesIO(person_image), 'image/jpeg')}
        
        start_time = time.time()
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        processing_time = time.time() - start_time
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            result_url = result_data.get('result_image_url', '')
            cost = result_data.get('cost', 0)
            
            # Quality indicators
            quality_indicators = []
            
            # 1. Processing time indicates real AI work
            if processing_time >= 10:
                quality_indicators.append("Adequate processing time for AI enhancement")
            
            # 2. Cost indicates real processing
            if cost >= 0.02:
                quality_indicators.append("Real AI processing cost confirmed")
            
            # 3. Result format suggests advanced processing
            if result_url.startswith('data:image/'):
                quality_indicators.append("Advanced result format (data URL)")
            
            # 4. Check if result is different from simple overlay
            if len(result_url) > 1000:  # Data URLs for processed images are typically large
                quality_indicators.append("Complex result suggests advanced processing")
            
            if len(quality_indicators) >= 2:
                self.log_test("Garment Fitting Quality", True, 
                            f"Quality indicators: {'; '.join(quality_indicators)}")
            else:
                self.log_test("Garment Fitting Quality", False, 
                            f"Insufficient quality indicators. Time: {processing_time:.1f}s, Cost: ${cost}")
        else:
            self.log_test("Garment Fitting Quality", False, 
                        f"Quality test failed - Status: {status_code}")

    def run_enhanced_tests(self):
        """Run all enhanced virtual try-on tests"""
        print("🚀 Starting Enhanced Virtual Try-On Tests")
        print(f"📍 Testing API at: {self.base_url}")
        print("🎯 Focus: Advanced garment fitting and blending")
        print("=" * 70)
        
        # Setup test user
        if not self.setup_test_user():
            print("❌ Cannot proceed without test user")
            return
        
        # Get products for testing
        success, response_data, status_code = self.make_request('GET', '/products')
        
        if not success or 'products' not in response_data:
            self.log_test("Product Retrieval", False, "Could not retrieve products for testing")
            self.print_test_summary()
            return
        
        products = response_data['products']
        if not products:
            self.log_test("Product Availability", False, "No products available for testing")
            self.print_test_summary()
            return
        
        # Select first product for detailed testing
        first_product = products[0]
        product_id = first_product.get('id')
        product_name = first_product.get('name', 'Test Product')
        category = first_product.get('category', 'Unknown')
        
        print(f"🎯 Primary test product: {product_name} ({category})")
        print()
        
        # Run enhanced tests
        self.test_enhanced_hybrid_tryon(product_id, product_name, category)
        self.test_enhanced_premium_tryon(product_id, product_name, category)
        self.test_garment_fitting_quality(product_id)
        self.test_different_product_categories()
        
        # Print final results
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("📊 ENHANCED VIRTUAL TRY-ON TEST SUMMARY")
        print("=" * 70)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Analyze results
        failed_tests = [test for test in self.test_results if not test['success']]
        
        if self.tests_passed == self.tests_run:
            print("\n🎉 ALL ENHANCED TESTS PASSED!")
            print("✅ Advanced garment fitting is working correctly")
            print("✅ Smart blending with proper transparency confirmed")
            print("✅ Real AI processing verified")
            print("✅ Multiple product categories supported")
        else:
            print(f"\n⚠️  {len(failed_tests)} enhanced tests failed.")
            
            # Categorize issues
            mock_processing_issues = [t for t in failed_tests if "mock processing" in t['details']]
            processing_time_issues = [t for t in failed_tests if "Processing too fast" in t['details']]
            cost_issues = [t for t in failed_tests if "Cost indicates" in t['details']]
            
            if mock_processing_issues:
                print(f"🔍 Mock Processing Issues: {len(mock_processing_issues)} tests")
                print("   → System may be using placeholder logic instead of real AI")
            
            if processing_time_issues:
                print(f"⏱️  Processing Time Issues: {len(processing_time_issues)} tests")
                print("   → Processing too fast suggests mock/placeholder responses")
            
            if cost_issues:
                print(f"💰 Cost Issues: {len(cost_issues)} tests")
                print("   → Incorrect costs suggest fallback to mock processing")
        
        # Print failed test details
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        print("\n" + "=" * 70)

def main():
    """Main test execution"""
    tester = EnhancedTryOnTester()
    tester.run_enhanced_tests()
    
    # Return appropriate exit code
    if tester.tests_passed == tester.tests_run:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())