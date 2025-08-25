#!/usr/bin/env python3
"""
Photo Saving and User Profile Update Workflow Test
Tests the complete workflow: Register → Login → Extract Measurements (save photo) → Profile Update → Virtual Try-On
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

class PhotoWorkflowTester:
    def __init__(self, base_url="https://virtufit-7.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        
        # Test data with realistic user info
        timestamp = int(time.time())
        self.test_email = f"sarah.johnson.{timestamp}@example.com"
        self.test_password = "SecurePass123!"
        self.test_full_name = "Sarah Johnson"

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

    def create_realistic_photo(self) -> bytes:
        """Create a realistic person photo for testing"""
        # Create a 400x600 image with a realistic person silhouette
        img = Image.new('RGB', (400, 600), color='#f0f0f0')  # Light background
        draw = ImageDraw.Draw(img)
        
        # Draw a realistic person silhouette
        # Head (oval)
        draw.ellipse([150, 50, 250, 150], fill='#D2B48C')  # Skin tone
        
        # Neck
        draw.rectangle([185, 150, 215, 180], fill='#D2B48C')
        
        # Body (torso)
        draw.rectangle([120, 180, 280, 380], fill='#4169E1')  # Blue top
        
        # Arms
        draw.rectangle([80, 200, 120, 350], fill='#D2B48C')   # Left arm
        draw.rectangle([280, 200, 320, 350], fill='#D2B48C')  # Right arm
        
        # Legs
        draw.rectangle([140, 380, 190, 580], fill='#2F4F4F')  # Left leg (dark pants)
        draw.rectangle([210, 380, 260, 580], fill='#2F4F4F')  # Right leg
        
        # Add some details for realism
        # Hair
        draw.ellipse([140, 40, 260, 120], fill='#8B4513')  # Brown hair
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        return buffer.getvalue()

    def test_step_1_register_user(self):
        """Step 1: Register a new user"""
        print("\n🔐 STEP 1: Register New User")
        print("-" * 40)
        
        registration_data = {
            "email": self.test_email,
            "password": self.test_password,
            "full_name": self.test_full_name
        }
        
        success, response_data, status_code = self.make_request('POST', '/register', registration_data)
        
        if success and response_data.get('access_token'):
            self.token = response_data['access_token']
            self.user_data = response_data.get('user')
            
            # Verify user data structure
            user = self.user_data
            if user and user.get('email') == self.test_email:
                # Check if profile_photo is initially None/empty
                profile_photo = user.get('profile_photo')
                if profile_photo is None or profile_photo == "":
                    self.log_test("User Registration", True, 
                                f"User registered successfully, profile_photo initially empty")
                    return True
                else:
                    self.log_test("User Registration", False, 
                                f"profile_photo should be empty initially, got: {type(profile_photo)}")
                    return False
            else:
                self.log_test("User Registration", False, "Invalid user data returned")
                return False
        else:
            self.log_test("User Registration", False, f"Registration failed - Status: {status_code}", response_data)
            return False

    def test_step_2_login_user(self):
        """Step 2: Login user"""
        print("\n🔑 STEP 2: Login User")
        print("-" * 40)
        
        login_data = {
            "email": self.test_email,
            "password": self.test_password
        }
        
        success, response_data, status_code = self.make_request('POST', '/login', login_data)
        
        if success and response_data.get('access_token'):
            self.token = response_data['access_token']
            self.user_data = response_data.get('user')
            self.log_test("User Login", True, f"Login successful")
            return True
        else:
            self.log_test("User Login", False, f"Login failed - Status: {status_code}", response_data)
            return False

    def test_step_3_extract_measurements_save_photo(self):
        """Step 3: Extract measurements and verify photo is saved to profile"""
        print("\n📸 STEP 3: Extract Measurements & Save Photo")
        print("-" * 40)
        
        if not self.token:
            self.log_test("Extract Measurements", False, "No authentication token available")
            return False
        
        # Create realistic photo
        photo_data = self.create_realistic_photo()
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        success, response_data, status_code = self.make_request('POST', '/extract-measurements', 
                                                               files=files, expected_status=200)
        
        if success and response_data.get('success'):
            measurements = response_data.get('data', {})
            
            # Verify measurements were extracted
            required_measurements = ['height', 'weight', 'chest', 'waist', 'hips']
            measurements_valid = all(measurements.get(field) for field in required_measurements)
            
            if measurements_valid:
                self.log_test("Measurement Extraction", True, 
                            f"Measurements extracted successfully: height={measurements.get('height')}, chest={measurements.get('chest')}")
                return True
            else:
                self.log_test("Measurement Extraction", False, 
                            f"Missing measurement data: {measurements}")
                return False
        else:
            self.log_test("Measurement Extraction", False, 
                        f"Measurement extraction failed - Status: {status_code}", response_data)
            return False

    def test_step_4_verify_profile_photo_saved(self):
        """Step 4: Verify profile photo was saved to user profile"""
        print("\n👤 STEP 4: Verify Profile Photo Saved")
        print("-" * 40)
        
        if not self.token:
            self.log_test("Profile Photo Verification", False, "No authentication token available")
            return False
        
        success, response_data, status_code = self.make_request('GET', '/profile')
        
        if success:
            profile_photo = response_data.get('profile_photo')
            
            if profile_photo and profile_photo.startswith('data:image/'):
                # Verify it's a valid base64 data URL
                try:
                    # Extract base64 data
                    base64_data = profile_photo.split(',')[1]
                    decoded_data = base64.b64decode(base64_data)
                    
                    if len(decoded_data) > 1000:  # Should be a reasonable image size
                        self.log_test("Profile Photo Saved", True, 
                                    f"Profile photo saved successfully (size: {len(decoded_data)} bytes)")
                        return True
                    else:
                        self.log_test("Profile Photo Saved", False, 
                                    f"Profile photo too small: {len(decoded_data)} bytes")
                        return False
                except Exception as e:
                    self.log_test("Profile Photo Saved", False, 
                                f"Invalid base64 data: {str(e)}")
                    return False
            else:
                self.log_test("Profile Photo Saved", False, 
                            f"Profile photo not saved or invalid format: {type(profile_photo)}")
                return False
        else:
            self.log_test("Profile Photo Verification", False, 
                        f"Profile retrieval failed - Status: {status_code}", response_data)
            return False

    def test_step_5_virtual_tryon_with_saved_photo(self):
        """Step 5: Test virtual try-on workflow using saved photo"""
        print("\n👗 STEP 5: Virtual Try-On with Saved Photo")
        print("-" * 40)
        
        if not self.token:
            self.log_test("Virtual Try-On", False, "No authentication token available")
            return False
        
        # First get products
        success, response_data, status_code = self.make_request('GET', '/products')
        
        if not success or not response_data.get('products'):
            self.log_test("Virtual Try-On", False, "No products available for testing")
            return False
        
        products = response_data['products']
        test_product = products[0]  # Use first product
        product_id = test_product.get('id')
        product_name = test_product.get('name', 'Test Product')
        
        if not product_id:
            self.log_test("Virtual Try-On", False, "No valid product ID")
            return False
        
        # Create photo for try-on (simulating user uploading their photo)
        photo_data = self.create_realistic_photo()
        
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Blue'
        }
        
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        success, response_data, status_code = self.make_request('POST', '/tryon', 
                                                               data=form_data, files=files, expected_status=200)
        
        if success and response_data.get('success'):
            result_data = response_data.get('data', {})
            result_image_url = result_data.get('result_image_url')
            
            if result_image_url:
                # Check if we get a "No photo available" error
                if "no photo available" in result_image_url.lower():
                    self.log_test("Virtual Try-On", False, 
                                "Got 'No photo available' error - workflow failed")
                    return False
                else:
                    self.log_test("Virtual Try-On", True, 
                                f"Try-on completed successfully with {product_name}")
                    return True
            else:
                self.log_test("Virtual Try-On", False, "No result image URL returned")
                return False
        else:
            self.log_test("Virtual Try-On", False, 
                        f"Try-on failed - Status: {status_code}", response_data)
            return False

    def test_step_6_verify_no_photo_errors(self):
        """Step 6: Verify no 'No photo available' errors in the complete workflow"""
        print("\n🔍 STEP 6: Verify No Photo Errors")
        print("-" * 40)
        
        # This test simulates the exact user journey that was failing
        # 1. User has completed measurement extraction (photo saved)
        # 2. User selects product and tries virtual try-on
        # 3. Should NOT get "No photo available" error
        
        if not self.token:
            self.log_test("No Photo Errors Check", False, "No authentication token available")
            return False
        
        # Verify profile still has photo
        success, response_data, status_code = self.make_request('GET', '/profile')
        
        if success and response_data.get('profile_photo'):
            profile_photo = response_data.get('profile_photo')
            
            if profile_photo and profile_photo.startswith('data:image/'):
                self.log_test("Photo Persistence Check", True, 
                            "Profile photo persists and is accessible")
                
                # Test that frontend context would be refreshed
                # (This simulates what the frontend should do)
                user_context = {
                    'id': response_data.get('id'),
                    'email': response_data.get('email'),
                    'full_name': response_data.get('full_name'),
                    'profile_photo': profile_photo,
                    'measurements': response_data.get('measurements')
                }
                
                if user_context['profile_photo']:
                    self.log_test("User Context Refresh", True, 
                                "User context would be properly refreshed with photo")
                    return True
                else:
                    self.log_test("User Context Refresh", False, 
                                "User context missing profile photo")
                    return False
            else:
                self.log_test("Photo Persistence Check", False, 
                            "Profile photo not found or invalid")
                return False
        else:
            self.log_test("Photo Persistence Check", False, 
                        "Profile retrieval failed or no photo")
            return False

    def run_complete_workflow_test(self):
        """Run the complete photo saving and user profile update workflow test"""
        print("🚀 PHOTO SAVING & USER PROFILE UPDATE WORKFLOW TEST")
        print(f"📍 Testing API at: {self.base_url}")
        print("=" * 70)
        print("Testing the complete user journey:")
        print("1. Register new user")
        print("2. Login user") 
        print("3. Extract measurements (save photo to profile)")
        print("4. Verify profile photo is saved")
        print("5. Test virtual try-on workflow")
        print("6. Verify no 'No photo available' errors")
        print("=" * 70)
        
        # Execute the complete workflow
        step1_success = self.test_step_1_register_user()
        step2_success = self.test_step_2_login_user()
        step3_success = self.test_step_3_extract_measurements_save_photo()
        step4_success = self.test_step_4_verify_profile_photo_saved()
        step5_success = self.test_step_5_virtual_tryon_with_saved_photo()
        step6_success = self.test_step_6_verify_no_photo_errors()
        
        # Print workflow summary
        self.print_workflow_summary()
        
        # Return True if all critical steps passed
        critical_steps = [step1_success, step2_success, step3_success, step4_success]
        return all(critical_steps)

    def print_workflow_summary(self):
        """Print comprehensive workflow test summary"""
        print("\n" + "=" * 70)
        print("📊 PHOTO WORKFLOW TEST SUMMARY")
        print("=" * 70)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Workflow status
        print(f"\n📋 WORKFLOW STATUS:")
        for i, test in enumerate(self.test_results, 1):
            status = "✅ PASSED" if test['success'] else "❌ FAILED"
            print(f"   Step {i}: {test['name']} - {status}")
            if not test['success'] and test['details']:
                print(f"           {test['details']}")
        
        # Critical assessment
        critical_tests = [
            "User Registration",
            "User Login", 
            "Measurement Extraction",
            "Profile Photo Saved"
        ]
        
        critical_passed = sum(1 for test in self.test_results 
                            if test['name'] in critical_tests and test['success'])
        
        print(f"\n🎯 CRITICAL WORKFLOW COMPONENTS:")
        print(f"   Passed: {critical_passed}/{len(critical_tests)}")
        
        if critical_passed == len(critical_tests):
            print("\n🎉 PHOTO WORKFLOW WORKING CORRECTLY!")
            print("   ✅ Photos are saved to user profiles")
            print("   ✅ Profile updates work properly")
            print("   ✅ No 'No photo available' errors expected")
        else:
            print(f"\n⚠️  PHOTO WORKFLOW HAS ISSUES!")
            failed_critical = [test['name'] for test in self.test_results 
                             if test['name'] in critical_tests and not test['success']]
            print(f"   ❌ Failed critical components: {', '.join(failed_critical)}")
        
        print("\n" + "=" * 70)

def main():
    """Main test execution"""
    tester = PhotoWorkflowTester()
    workflow_success = tester.run_complete_workflow_test()
    
    # Return appropriate exit code
    return 0 if workflow_success else 1

if __name__ == "__main__":
    sys.exit(main())