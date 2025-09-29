#!/usr/bin/env python3
"""
FASHN API Test Script
Tests the correct FASHN virtual try-on API
"""

import asyncio
import aiohttp
import base64
import json
from PIL import Image
import io

# FASHN API Configuration
FASHN_API_KEY = "fa-hfuTx6vji1ma-KeeQTDToyTwCObNZrPbtnO9w"
FASHN_BASE_URL = "https://api.fashn.ai/v1"

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

async def test_fashn_api():
    """Test FASHN virtual try-on API"""
    print("Testing FASHN API...")
    
    # Create test images
    print("Creating test images...")
    customer_img = create_test_image('lightblue', text="PERSON")
    garment_img = create_test_image('red', text="SHIRT")
    
    # Convert to base64
    customer_b64 = image_to_base64(customer_img)
    garment_b64 = image_to_base64(garment_img)
    
    # FASHN API payload
    payload = {
        "model_name": "product-to-model",
        "inputs": {
            "product_image": f"data:image/jpeg;base64,{garment_b64}",
            "model_image": f"data:image/jpeg;base64,{customer_b64}",
            "output_format": "png",
            "return_base64": True
        }
    }
    
    headers = {
        "Authorization": f"Bearer {FASHN_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Calling FASHN API: {FASHN_BASE_URL}/run")
    print(f"API Key: {FASHN_API_KEY[:20]}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Submit job
            async with session.post(
                f"{FASHN_BASE_URL}/run",
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                
                print(f"Submit Response Status: {response.status}")
                
                if response.status == 200:
                    job_result = await response.json()
                    job_id = job_result.get("id")
                    print(f"Job ID: {job_id}")
                    
                    if job_id:
                        # Poll for completion
                        result = await poll_job_status(session, job_id, headers)
                        return result
                    else:
                        return {"success": False, "error": "No job ID returned"}
                else:
                    error_text = await response.text()
                    print(f"Submit Error: {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                    
    except Exception as e:
        print(f"FASHN API Exception: {str(e)}")
        return {"success": False, "error": str(e)}

async def poll_job_status(session, job_id, headers, max_attempts=30):
    """Poll FASHN job status"""
    print(f"Polling job status for {job_id}...")
    
    for attempt in range(max_attempts):
        try:
            async with session.get(
                f"{FASHN_BASE_URL}/status/{job_id}",
                headers=headers,
                timeout=10
            ) as response:
                
                if response.status == 200:
                    status_result = await response.json()
                    status = status_result.get("status")
                    print(f"Attempt {attempt + 1}: Status = {status}")
                    
                    if status == "completed":
                        output = status_result.get("output", [])
                        if output:
                            result_image = output[0]
                            print(f"Job completed! Result: {result_image[:50]}...")
                            
                            # Handle base64 or URL
                            if result_image.startswith("data:image"):
                                result_b64 = result_image.split(",")[1]
                            else:
                                # Download from URL
                                async with session.get(result_image) as img_response:
                                    img_bytes = await img_response.read()
                                    result_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            
                            # Save result
                            result_bytes = base64.b64decode(result_b64)
                            result_img = Image.open(io.BytesIO(result_bytes))
                            result_img.save("fashn_test_result.png")
                            print("Result saved as 'fashn_test_result.png'")
                            
                            return {
                                "success": True,
                                "result_image_base64": result_b64,
                                "processing_time": attempt * 2
                            }
                        else:
                            return {"success": False, "error": "No output image"}
                    
                    elif status == "failed":
                        error = status_result.get("error", "Unknown error")
                        return {"success": False, "error": f"Job failed: {error}"}
                    
                    elif status in ["pending", "processing"]:
                        await asyncio.sleep(2)
                        continue
                    
                    else:
                        return {"success": False, "error": f"Unknown status: {status}"}
                
                else:
                    print(f"Status check failed: {response.status}")
                    await asyncio.sleep(2)
        
        except Exception as e:
            print(f"Polling error: {e}")
            if attempt == max_attempts - 1:
                return {"success": False, "error": f"Polling failed: {e}"}
            await asyncio.sleep(2)
    
    return {"success": False, "error": "Polling timeout"}

async def main():
    """Main test function"""
    print("FASHN API Test Suite")
    print("=" * 40)
    
    result = await test_fashn_api()
    
    print("\nTest Summary:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Processing Time: {result.get('processing_time', 'unknown')}s")
    else:
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result['success'] else 1)