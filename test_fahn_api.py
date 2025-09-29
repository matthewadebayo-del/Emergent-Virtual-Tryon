#!/usr/bin/env python3
"""
FAHN API Test Script
Tests the FAHN virtual try-on API with sample images
"""

import asyncio
import aiohttp
import base64
import json
from PIL import Image
import io
import os

# FAHN API Configuration
FAHN_API_KEY = "fa-hfuTx6vji1ma-KeeQTDToyTwCObNZrPbtnO9w"
FAHN_BASE_URL = "https://api.fahn.ai/v1"

def create_test_image(color, size=(512, 512), text=""):
    """Create a test image with specified color"""
    img = Image.new('RGB', size, color=color)
    if text:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        draw.text((size[0]//4, size[1]//2), text, fill='white', font=font)
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

async def test_fahn_api():
    """Test FAHN virtual try-on API"""
    print("Testing FAHN API...")
    
    # Create test images
    print("Creating test images...")
    customer_img = create_test_image('lightblue', text="CUSTOMER")
    garment_img = create_test_image('red', text="T-SHIRT")
    
    # Convert to base64
    customer_b64 = image_to_base64(customer_img)
    garment_b64 = image_to_base64(garment_img)
    
    # Prepare API request
    payload = {
        "model": "virtual-tryon-v1",
        "customer_image": customer_b64,
        "garment_image": garment_b64,
        "product_info": {
            "name": "Red T-Shirt",
            "category": "shirts",
            "color": "red"
        },
        "options": {
            "preserve_identity": True,
            "quality": "high"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {FAHN_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Calling FAHN API: {FAHN_BASE_URL}/virtual-tryon")
    print(f"API Key: {FAHN_API_KEY[:20]}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{FAHN_BASE_URL}/virtual-tryon",
                json=payload,
                headers=headers,
                timeout=60
            ) as response:
                
                print(f"Response Status: {response.status}")
                print(f"Response Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print("FAHN API Success!")
                    print(f"Result keys: {list(result.keys())}")
                    
                    if "result_image" in result:
                        # Save result image
                        result_b64 = result["result_image"]
                        result_bytes = base64.b64decode(result_b64)
                        result_img = Image.open(io.BytesIO(result_bytes))
                        result_img.save("fahn_test_result.jpg")
                        print("Result image saved as 'fahn_test_result.jpg'")
                    
                    return {
                        "success": True,
                        "response": result,
                        "processing_time": result.get("processing_time", "unknown")
                    }
                else:
                    error_text = await response.text()
                    print(f"FAHN API Error {response.status}")
                    print(f"Error Response: {error_text}")
                    
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "status_code": response.status
                    }
                    
    except asyncio.TimeoutError:
        print("FAHN API Timeout")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        print(f"FAHN API Exception: {str(e)}")
        return {"success": False, "error": str(e)}

async def test_fahn_endpoints():
    """Test different FAHN API endpoints"""
    print("Testing FAHN API endpoints...")
    
    # Test health/status endpoint
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{FAHN_BASE_URL}/health") as response:
                print(f"Health endpoint: {response.status}")
    except Exception as e:
        print(f"Health endpoint error: {str(e)}")
    
    # Test models endpoint
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {FAHN_API_KEY}"}
            async with session.get(f"{FAHN_BASE_URL}/models", headers=headers) as response:
                print(f"Models endpoint: {response.status}")
                if response.status == 200:
                    models = await response.json()
                    print(f"Available models: {models}")
    except Exception as e:
        print(f"Models endpoint error: {str(e)}")

async def main():
    """Main test function"""
    print("FAHN API Test Suite")
    print("=" * 50)
    
    # Test API endpoints
    await test_fahn_endpoints()
    print()
    
    # Test virtual try-on
    result = await test_fahn_api()
    print()
    
    # Summary
    print("Test Summary:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Processing Time: {result.get('processing_time', 'unknown')}")
    else:
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if result['success'] else 1)