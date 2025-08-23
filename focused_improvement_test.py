#!/usr/bin/env python3
"""
Focused test for Virtual Try-On improvements based on user feedback:
1. Measurement Units (inches/pounds instead of cm/kg)
2. Size Recommendations (XS to XXXL)
3. Virtual Try-On appearance preservation
4. User notifications about limitations
"""

import requests
import json
import sys
from datetime import datetime

class ImprovementTester:
    def __init__(self, base_url="https://virtual-tryon-app.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")
        return success

    def setup_user(self):
        """Create and authenticate a test user"""
        timestamp = datetime.now().strftime('%H%M%S')
        email = f"test_improvements_{timestamp}@example.com"
        password = "TestPass123!"
        
        # Register user
        response = requests.post(f"{self.base_url}/register", json={
            "email": email,
            "password": password,
            "full_name": f"Test User {timestamp}"
        })
        
        if response.status_code == 200:
            self.token = response.json()['access_token']
            print(f"✅ User setup complete: {email}")
            return True
        else:
            print(f"❌ User setup failed: {response.status_code}")
            return False

    def test_measurement_units_conversion(self):
        """Test that measurements are returned in inches/pounds"""
        print("\n🔍 Testing Measurement Units Conversion...")
        
        if not self.token:
            return self.log_test("Measurement Units", False, "No auth token")
        
        # Create test image (1x1 pixel PNG)
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {'user_image_base64': test_image}
        
        response = requests.post(f"{self.base_url}/extract-measurements", 
                               data=form_data, headers=headers)
        
        if response.status_code != 200:
            return self.log_test("Measurement Units", False, f"API error: {response.status_code}")
        
        data = response.json()
        measurements = data.get('measurements', {})
        
        # Check if measurements are in inches (height should be reasonable for inches)
        height = measurements.get('height', 0)
        weight = measurements.get('weight', 0)
        
        # Height in inches should be roughly 60-80 for adults
        # Weight in pounds should be roughly 100-300 for adults
        height_in_inches = 60 <= height <= 80
        weight_in_pounds = 100 <= weight <= 300
        
        details = f"Height: {height} (inches: {height_in_inches}), Weight: {weight} (pounds: {weight_in_pounds})"
        success = height_in_inches and weight_in_pounds
        
        return self.log_test("Measurement Units", success, details)

    def test_enhanced_size_recommendations(self):
        """Test that size recommendations include XS to XXXL range"""
        print("\n🔍 Testing Enhanced Size Recommendations...")
        
        if not self.token:
            return self.log_test("Size Recommendations", False, "No auth token")
        
        # Get products first
        response = requests.get(f"{self.base_url}/products")
        if response.status_code != 200:
            return self.log_test("Size Recommendations", False, "Cannot get products")
        
        products = response.json()
        if not products:
            return self.log_test("Size Recommendations", False, "No products available")
        
        # Test virtual try-on to get size recommendation
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {
            'user_image_base64': test_image,
            'product_id': products[0]['id'],
            'use_stored_measurements': 'true'
        }
        
        response = requests.post(f"{self.base_url}/tryon", 
                               data=form_data, headers=headers)
        
        if response.status_code != 200:
            return self.log_test("Size Recommendations", False, f"Try-on failed: {response.status_code}")
        
        data = response.json()
        size_rec = data.get('size_recommendation', '')
        
        # Check if size is in the enhanced range (XS, S, M, L, XL, XXL, XXXL)
        valid_sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']
        success = size_rec in valid_sizes
        
        details = f"Recommended size: {size_rec} (valid: {success})"
        return self.log_test("Size Recommendations", success, details)

    def test_virtual_tryon_enhancements(self):
        """Test virtual try-on with enhanced appearance preservation"""
        print("\n🔍 Testing Virtual Try-On Enhancements...")
        
        if not self.token:
            return self.log_test("Virtual Try-On Enhancements", False, "No auth token")
        
        # Get products
        response = requests.get(f"{self.base_url}/products")
        if response.status_code != 200:
            return self.log_test("Virtual Try-On Enhancements", False, "Cannot get products")
        
        products = response.json()
        if not products:
            return self.log_test("Virtual Try-On Enhancements", False, "No products available")
        
        # Test virtual try-on
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {
            'user_image_base64': test_image,
            'product_id': products[0]['id'],
            'use_stored_measurements': 'false'
        }
        
        response = requests.post(f"{self.base_url}/tryon", 
                               data=form_data, headers=headers)
        
        if response.status_code != 200:
            return self.log_test("Virtual Try-On Enhancements", False, f"Try-on failed: {response.status_code}")
        
        data = response.json()
        
        # Check for enhanced features
        has_result_image = 'result_image_base64' in data and len(data['result_image_base64']) > 0
        has_measurements = 'measurements_used' in data
        has_size_rec = 'size_recommendation' in data
        has_personalization_note = 'personalization_note' in data
        
        success = has_result_image and has_measurements and has_size_rec
        
        details = f"Image: {has_result_image}, Measurements: {has_measurements}, Size: {has_size_rec}, Note: {has_personalization_note}"
        return self.log_test("Virtual Try-On Enhancements", success, details)

    def test_user_notifications(self):
        """Test that user notifications about limitations are present"""
        print("\n🔍 Testing User Notifications...")
        
        if not self.token:
            return self.log_test("User Notifications", False, "No auth token")
        
        # Test virtual try-on to check for limitation notices
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        headers = {'Authorization': f'Bearer {self.token}'}
        form_data = {
            'user_image_base64': test_image,
            'use_stored_measurements': 'false'
        }
        
        response = requests.post(f"{self.base_url}/tryon", 
                               data=form_data, headers=headers)
        
        if response.status_code != 200:
            return self.log_test("User Notifications", False, f"Try-on failed: {response.status_code}")
        
        data = response.json()
        
        # Check for personalization note that explains limitations
        personalization_note = data.get('personalization_note', '')
        has_limitation_info = len(personalization_note) > 0
        
        details = f"Personalization note present: {has_limitation_info}"
        if has_limitation_info:
            details += f" (Note: {personalization_note[:100]}...)"
        
        return self.log_test("User Notifications", has_limitation_info, details)

    def run_all_tests(self):
        """Run all improvement tests"""
        print("🚀 Testing Virtual Try-On Improvements")
        print("=" * 60)
        
        # Setup
        if not self.setup_user():
            print("❌ Cannot proceed without user setup")
            return False
        
        # Run focused tests
        test1 = self.test_measurement_units_conversion()
        test2 = self.test_enhanced_size_recommendations()
        test3 = self.test_virtual_tryon_enhancements()
        test4 = self.test_user_notifications()
        
        # Results
        print("\n" + "=" * 60)
        print(f"📊 Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All improvement tests passed!")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed")
            return False

def main():
    tester = ImprovementTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())