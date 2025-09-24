import requests
import base64
import io
from PIL import Image, ImageDraw

def create_test_image(color='blue'):
    img = Image.new('RGB', (512, 512), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([128, 128, 384, 384], outline='white', width=3)
    draw.text((200, 250), "TEST", fill='white')
    
    with io.BytesIO() as buffer:
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

print("Testing Production Virtual Try-On System...")

# Test 1: Server Health
print("\n1. Testing server health...")
response = requests.get("http://localhost:8002/health")
print(f"Health Status: {response.status_code}")
if response.status_code == 200:
    print(f"Response: {response.json()}")

# Test 2: Server Capabilities  
print("\n2. Testing capabilities...")
response = requests.get("http://localhost:8002/debug")
print(f"Debug Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Full 3D Pipeline: {data['capabilities']['full_3d_pipeline']}")
    print(f"AI Enhancement: {data['capabilities']['ai_enhancement']}")

# Test 3: Virtual Try-On (without auth for now)
print("\n3. Testing virtual try-on...")
user_image = create_test_image('lightblue')
garment_image = create_test_image('red')

tryon_data = {
    "user_image_base64": user_image,
    "garment_image_base64": garment_image,
    "processing_mode": "fallback"
}

# Try without auth first to see what happens
response = requests.post("http://localhost:8002/api/virtual-tryon", data=tryon_data)
print(f"Try-on Status: {response.status_code}")
print(f"Response: {response.text[:200]}...")

print("\nTest complete!")