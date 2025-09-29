#!/usr/bin/env python3
"""
Test Vertex AI Virtual Try-On API
Tests Matthew wearing the green t-shirt
"""

import base64
import requests
import json
import subprocess
import os

PROJECT_ID = "virtual-tryon-solution"

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_access_token():
    """Get Google Cloud access token"""
    try:
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get access token: {e}")
        return None

def save_result_image(base64_data, filename="result.jpg"):
    """Save base64 image data to file"""
    try:
        image_data = base64.b64decode(base64_data)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"✅ Result saved as {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to save result: {e}")
        return False

def test_vertex_virtual_tryon():
    """Test Vertex AI virtual try-on with Matthew and green t-shirt"""
    
    print("🧪 Testing Vertex AI Virtual Try-On...")
    print("👤 Person: Matthew")
    print("👕 Clothing: Green T-Shirt")
    print()
    
    # Check if image files exist
    person_path = "person.jpg"
    clothing_path = "clothing.jpg"
    
    if not os.path.exists(person_path):
        print(f"❌ Person image not found: {person_path}")
        return False
        
    if not os.path.exists(clothing_path):
        print(f"❌ Clothing image not found: {clothing_path}")
        return False
    
    # Get access token
    print("🔑 Getting access token...")
    access_token = get_access_token()
    if not access_token:
        return False
    
    # Convert images to base64
    print("📸 Converting images to base64...")
    try:
        person_b64 = image_to_base64(person_path)
        clothing_b64 = image_to_base64(clothing_path)
        print(f"   Person image: {len(person_b64)} characters")
        print(f"   Clothing image: {len(clothing_b64)} characters")
    except Exception as e:
        print(f"❌ Failed to convert images: {e}")
        return False
    
    # Prepare API request
    print("🚀 Calling Vertex AI API...")
    
    payload = {
        "instances": [{
            "personImage": {
                "image": {
                    "bytesBase64Encoded": person_b64
                }
            },
            "productImages": [{
                "image": {
                    "bytesBase64Encoded": clothing_b64
                }
            }]
        }],
        "parameters": {
            "baseSteps": 32,
            "imageCount": 1
        }
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/virtual-try-on-preview-08-04:predict"
    
    try:
        print(f"   URL: {url}")
        print(f"   Payload size: {len(json.dumps(payload))} bytes")
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Virtual try-on successful!")
            
            predictions = result.get('predictions', [])
            print(f"📊 Generated {len(predictions)} images")
            
            if predictions and len(predictions) > 0:
                # Save the first result
                prediction = predictions[0]
                
                # The response structure might vary, check common fields
                image_data = None
                if 'generatedImage' in prediction:
                    if 'image' in prediction['generatedImage']:
                        if 'bytesBase64Encoded' in prediction['generatedImage']['image']:
                            image_data = prediction['generatedImage']['image']['bytesBase64Encoded']
                
                if image_data:
                    if save_result_image(image_data, "matthew_green_tshirt_result.jpg"):
                        print("🎉 Test completed successfully!")
                        print("📁 Check 'matthew_green_tshirt_result.jpg' to see Matthew wearing the green t-shirt")
                        return True
                else:
                    print("⚠️  Response received but no image data found")
                    print("📋 Response structure:")
                    print(json.dumps(result, indent=2)[:500] + "...")
                    return False
            else:
                print("⚠️  No predictions in response")
                return False
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (60 seconds)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("🧪 VERTEX AI VIRTUAL TRY-ON TEST")
    print("=" * 50)
    
    success = test_vertex_virtual_tryon()
    
    print()
    print("=" * 50)
    if success:
        print("✅ TEST PASSED - Vertex AI virtual try-on works!")
        print("📸 Check the result image to evaluate quality")
    else:
        print("❌ TEST FAILED - Vertex AI virtual try-on not working")
        print("💡 Consider using FAHN as primary service")
    print("=" * 50)

if __name__ == "__main__":
    main()