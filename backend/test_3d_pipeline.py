#!/usr/bin/env python3
"""
Test script for the complete 3D virtual try-on pipeline
"""

import sys
import os
import tempfile
import base64
from PIL import Image
import numpy as np

sys.path.insert(0, '/home/ubuntu/repos/Emergent-Virtual-Tryon/backend')

from src.core.body_reconstruction import BodyReconstructor
from src.core.garment_fitting import GarmentFitter
from src.core.rendering import PhotorealisticRenderer
from src.core.ai_enhancement import AIEnhancer

def create_test_image():
    """Create a simple test image for testing"""
    img = Image.new('RGB', (512, 512), color='white')
    pixels = np.array(img)
    
    height, width = pixels.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    for y in range(center_y - 100, center_y - 50):
        for x in range(center_x - 25, center_x + 25):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Skin color
    
    for y in range(center_y - 50, center_y + 100):
        for x in range(center_x - 40, center_x + 40):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Skin color
    
    for y in range(center_y - 30, center_y + 50):
        for x in range(center_x - 70, center_x - 40):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Left arm
        for x in range(center_x + 40, center_x + 70):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Right arm
    
    for y in range(center_y + 100, center_y + 200):
        for x in range(center_x - 20, center_x):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Left leg
        for x in range(center_x, center_x + 20):
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = [200, 180, 160]  # Right leg
    
    return Image.fromarray(pixels)

def test_3d_pipeline():
    """Test the complete 3D virtual try-on pipeline"""
    print("🧪 Testing 3D Virtual Try-On Pipeline")
    print("=" * 50)
    
    try:
        print("1. Initializing modules...")
        body_reconstructor = BodyReconstructor()
        garment_fitter = GarmentFitter()
        renderer = PhotorealisticRenderer()
        ai_enhancer = AIEnhancer()
        print("✅ All modules initialized successfully")
        
        print("\n2. Creating test image...")
        test_image = create_test_image()
        
        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        image_bytes = img_bytes.getvalue()
        print("✅ Test image created")
        
        # Stage 1: Body Reconstruction
        print("\n3. Testing body reconstruction...")
        body_result = body_reconstructor.process_image_bytes(image_bytes)
        body_mesh = body_result['body_mesh']
        measurements = body_result['measurements']
        print(f"✅ Body mesh created with {len(body_mesh.vertices)} vertices")
        print(f"   Measurements: {measurements}")
        
        # Stage 2: Garment Fitting
        print("\n4. Testing garment fitting...")
        fitted_garment = garment_fitter.fit_garment_to_body(
            body_mesh, "shirts", "t_shirt"
        )
        print(f"✅ Garment fitted with {len(fitted_garment.vertices)} vertices")
        
        # Stage 3: Rendering
        print("\n5. Testing rendering...")
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            rendered_path = renderer.render_scene(
                body_mesh, fitted_garment, temp_file.name,
                fabric_type="cotton",
                fabric_color=(0.2, 0.3, 0.8)
            )
            
            if os.path.exists(rendered_path):
                file_size = os.path.getsize(rendered_path)
                print(f"✅ Rendering complete: {rendered_path} ({file_size} bytes)")
            else:
                print("⚠️ Rendered file not found")
        
        # Stage 4: AI Enhancement
        print("\n6. Testing AI enhancement...")
        rendered_image = Image.open(rendered_path)
        enhanced_image = ai_enhancer.enhance_realism(rendered_image, test_image)
        print(f"✅ AI enhancement complete: {enhanced_image.size}")
        
        print("\n" + "=" * 50)
        print("🎉 3D Virtual Try-On Pipeline Test SUCCESSFUL!")
        print("All modules are working correctly and can process images end-to-end.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_3d_pipeline()
    sys.exit(0 if success else 1)
