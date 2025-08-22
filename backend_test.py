import requests
import sys
import json
import base64
from datetime import datetime

class VirtualTryOnAPITester:
    def __init__(self, base_url="https://virtual-tryon-app.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.user_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        test_headers = {'Content-Type': 'application/json'}
        
        if self.token:
            test_headers['Authorization'] = f'Bearer {self.token}'
        
        if headers:
            test_headers.update(headers)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_register(self, email, password, full_name):
        """Test user registration"""
        success, response = self.run_test(
            "User Registration",
            "POST",
            "register",
            200,
            data={
                "email": email,
                "password": password,
                "full_name": full_name
            }
        )
        if success and 'access_token' in response:
            self.token = response['access_token']
            print(f"   Token received: {self.token[:20]}...")
            return True
        return False

    def test_login(self, email, password):
        """Test user login"""
        success, response = self.run_test(
            "User Login",
            "POST",
            "login",
            200,
            data={
                "email": email,
                "password": password
            }
        )
        if success and 'access_token' in response:
            self.token = response['access_token']
            print(f"   Token received: {self.token[:20]}...")
            return True
        return False

    def test_get_profile(self):
        """Test get user profile"""
        success, response = self.run_test(
            "Get User Profile",
            "GET",
            "profile",
            200
        )
        if success and 'id' in response:
            self.user_id = response['id']
            print(f"   User ID: {self.user_id}")
        return success

    def test_save_measurements(self):
        """Test save measurements"""
        measurements_data = {
            "height": 175.0,
            "weight": 70.0,
            "chest": 95.0,
            "waist": 80.0,
            "hips": 100.0,
            "shoulder_width": 45.0
        }
        success, response = self.run_test(
            "Save Measurements",
            "POST",
            "measurements",
            200,
            data=measurements_data
        )
        return success

    def test_get_products(self):
        """Test get products catalog"""
        success, response = self.run_test(
            "Get Products Catalog",
            "GET",
            "products",
            200
        )
        if success and isinstance(response, list) and len(response) > 0:
            print(f"   Found {len(response)} products")
            return True, response
        return False, []

    def test_virtual_tryon(self, product_id=None):
        """Test virtual try-on functionality using FormData (FIXED VERSION)"""
        # Create a simple base64 encoded test image (1x1 pixel PNG)
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # Use form data instead of JSON (THIS IS THE CRITICAL FIX)
        headers = {'Authorization': f'Bearer {self.token}'}
        url = f"{self.base_url}/tryon"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Virtual Try-On (FormData)...")
        print(f"   URL: {url}")
        
        try:
            # Send as form data (not JSON)
            form_data = {
                'user_image_base64': test_image_base64,
                'use_stored_measurements': 'true'
            }
            
            if product_id:
                form_data["product_id"] = product_id
                print(f"   Using product ID: {product_id}")
            
            response = requests.post(url, data=form_data, headers=headers)
            
            success = response.status_code == 200
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys())}")
                    if 'size_recommendation' in response_data:
                        print(f"   Size recommendation: {response_data['size_recommendation']}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_tryon_history(self):
        """Test get try-on history"""
        success, response = self.run_test(
            "Get Try-On History",
            "GET",
            "tryon-history",
            200
        )
        return success

    def test_extract_measurements(self):
        """Test extract measurements from image"""
        # Create a simple base64 encoded test image (1x1 pixel PNG)
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # Use form data for this endpoint
        headers = {'Authorization': f'Bearer {self.token}'}
        url = f"{self.base_url}/extract-measurements"
        
        self.tests_run += 1
        print(f"\n🔍 Testing Extract Measurements...")
        print(f"   URL: {url}")
        
        try:
            # Send as form data
            form_data = {'user_image_base64': test_image_base64}
            response = requests.post(url, data=form_data, headers=headers)
            
            success = response.status_code == 200
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_invalid_login(self):
        """Test login with invalid credentials"""
        success, response = self.run_test(
            "Invalid Login Test",
            "POST",
            "login",
            401,
            data={
                "email": "invalid@test.com",
                "password": "wrongpassword"
            }
        )
        return success

    def test_unauthorized_access(self):
        """Test accessing protected endpoint without token"""
        original_token = self.token
        self.token = None
        
        success, response = self.run_test(
            "Unauthorized Access Test",
            "GET",
            "profile",
            401
        )
        
        self.token = original_token
        return success

def main():
    print("🚀 Starting Virtual Try-On API Tests")
    print("=" * 50)
    
    # Setup
    tester = VirtualTryOnAPITester()
    test_timestamp = datetime.now().strftime('%H%M%S')
    test_email = f"test_user_{test_timestamp}@example.com"
    test_password = "TestPass123!"
    test_name = f"Test User {test_timestamp}"

    # Test 1: Health Check
    if not tester.test_health_check():
        print("❌ Health check failed, stopping tests")
        return 1

    # Test 2: User Registration
    if not tester.test_register(test_email, test_password, test_name):
        print("❌ Registration failed, stopping tests")
        return 1

    # Test 3: Get Profile (after registration)
    if not tester.test_get_profile():
        print("❌ Profile retrieval failed")
        return 1

    # Test 4: Save Measurements
    if not tester.test_save_measurements():
        print("❌ Save measurements failed")

    # Test 5: Get Products
    success, products = tester.test_get_products()
    if not success:
        print("❌ Get products failed")
        return 1

    # Test 6: Virtual Try-On (with first product if available)
    product_id = products[0]['id'] if products else None
    success, tryon_response = tester.test_virtual_tryon(product_id)
    if not success:
        print("❌ Virtual try-on failed - THIS IS THE CRITICAL TEST FOR 422 ERROR FIX")
        return 1
    else:
        print("✅ Virtual try-on succeeded - 422 error appears to be fixed!")

    # Test 7: Get Try-On History
    if not tester.test_tryon_history():
        print("❌ Try-on history failed")

    # Test 8: Extract Measurements (New Feature)
    if not tester.test_extract_measurements():
        print("❌ Extract measurements failed")

    # Test 9: Test new login with same credentials
    if not tester.test_login(test_email, test_password):
        print("❌ Login with registered credentials failed")

    # Test 10: Invalid Login
    if not tester.test_invalid_login():
        print("❌ Invalid login test failed")

    # Test 11: Unauthorized Access
    if not tester.test_unauthorized_access():
        print("❌ Unauthorized access test failed")

    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())