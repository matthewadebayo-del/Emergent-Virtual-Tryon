#!/usr/bin/env python3

import requests
import json
from datetime import datetime

def test_virtual_tryon_with_fresh_key():
    """Test virtual try-on with the fresh Universal Key"""
    base_url = "http://localhost:8000/api"
    
    print("🔑 Testing Virtual Try-On with Fresh Universal Key")
    print("=" * 60)
    
    # Step 1: Register a test user
    test_timestamp = datetime.now().strftime('%H%M%S')
    test_email = f"tryon_test_{test_timestamp}@example.com"
    test_password = "TestPass123!"
    test_name = f"TryOn Test User {test_timestamp}"
    
    print(f"📝 Registering test user: {test_email}")
    
    register_response = requests.post(f"{base_url}/register", json={
        "email": test_email,
        "password": test_password,
        "full_name": test_name
    })
    
    if register_response.status_code != 200:
        print(f"❌ Registration failed: {register_response.status_code}")
        return False
    
    token = register_response.json()['access_token']
    print(f"✅ Registration successful, token: {token[:20]}...")
    
    # Step 2: Get products to use one for try-on
    print(f"📦 Getting products...")
    
    products_response = requests.get(f"{base_url}/products", 
                                   headers={'Authorization': f'Bearer {token}'})
    
    if products_response.status_code != 200:
        print(f"❌ Failed to get products: {products_response.status_code}")
        return False
    
    products = products_response.json()
    if not products:
        print("❌ No products available")
        return False
    
    product_id = products[0]['id']
    product_name = products[0]['name']
    print(f"✅ Using product: {product_name} (ID: {product_id})")
    
    # Step 3: Test Virtual Try-On
    print(f"🎭 Testing Virtual Try-On with fresh key...")
    
    # Create a simple base64 encoded test image (1x1 pixel PNG)
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Use form data (not JSON) as the backend expects FormData
    form_data = {
        'user_image_base64': test_image_base64,
        'product_id': product_id,
        'use_stored_measurements': 'false'
    }
    
    headers = {'Authorization': f'Bearer {token}'}
    
    tryon_response = requests.post(f"{base_url}/tryon", 
                                 data=form_data, 
                                 headers=headers)
    
    print(f"📊 Try-on response status: {tryon_response.status_code}")
    
    if tryon_response.status_code == 200:
        print("✅ SUCCESS! Virtual Try-On completed successfully!")
        response_data = tryon_response.json()
        print(f"   Size recommendation: {response_data.get('size_recommendation', 'N/A')}")
        print(f"   Result image length: {len(response_data.get('result_image_base64', ''))}")
        print("🎉 Fresh Universal Key is working - budget issue resolved!")
        return True
    else:
        print(f"❌ FAILED! Virtual Try-On failed with status: {tryon_response.status_code}")
        try:
            error_data = tryon_response.json()
            print(f"   Error details: {error_data}")
        except:
            print(f"   Error text: {tryon_response.text}")
        
        if "budget" in tryon_response.text.lower():
            print("💰 Budget issue still exists - fresh key may not be loaded properly")
        
        return False

if __name__ == "__main__":
    success = test_virtual_tryon_with_fresh_key()
    exit(0 if success else 1)
