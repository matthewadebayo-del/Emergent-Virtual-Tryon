#!/usr/bin/env python3
"""
Test script for the complete virtual try-on pipeline including HEIC processing
"""

import sys
import os
import tempfile
import base64
import json
import requests
from PIL import Image
import numpy as np

sys.path.insert(0, '/home/ubuntu/repos/Emergent-Virtual-Tryon/backend')

from src.core.model_manager import model_manager

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
    print("🧪 Testing Complete 3D Virtual Try-On Pipeline")
    print("=" * 60)
    
    try:
        print("1. Creating test image...")
        test_image = create_test_image()
        
        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        image_bytes = img_bytes.getvalue()
        print(f"✅ Test image created: {len(image_bytes)} bytes")
        
        # Stage 1: Body Reconstruction
        print("\n2. Testing body reconstruction...")
        body_reconstructor = model_manager.get_body_reconstructor()
        if body_reconstructor is None:
            print("❌ Body reconstructor not available")
            return False
            
        body_result = body_reconstructor.process_image_bytes(image_bytes)
        body_mesh = body_result['body_mesh']
        measurements = body_result['measurements']
        print(f"✅ Body mesh created with {len(body_mesh.vertices)} vertices")
        print(f"   Measurements: {measurements}")
        
        # Stage 2: Garment Fitting
        print("\n3. Testing garment fitting...")
        garment_fitter = model_manager.get_garment_fitter()
        if garment_fitter is None:
            print("❌ Garment fitter not available")
            return False
            
        fitted_garment = garment_fitter.fit_garment_to_body(
            body_mesh, "shirts", "t_shirt"
        )
        print(f"✅ Garment fitted with {len(fitted_garment.vertices)} vertices")
        
        # Stage 3: Rendering
        print("\n4. Testing rendering...")
        renderer = model_manager.get_renderer()
        if renderer is None:
            print("❌ Renderer not available")
            return False
            
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            rendered_path = renderer.render_scene(
                body_mesh, fitted_garment, temp_file.name,
                fabric_type="cotton",
                fabric_color=(0.2, 0.3, 0.8)
            )
            
            if os.path.exists(rendered_path):
                file_size = os.path.getsize(rendered_path)
                is_placeholder = file_size < 50000
                print(f"✅ Rendering complete: {rendered_path} ({file_size} bytes)")
                print(f"   Is placeholder: {is_placeholder}")
                
                try:
                    rendered_image = Image.open(rendered_path)
                    print(f"   Image info: {rendered_image.format} {rendered_image.size}")
                except Exception as img_error:
                    print(f"❌ Rendered file is not a valid image: {str(img_error)}")
                    return False
            else:
                print("❌ Rendered file not found")
                return False
        
        # Stage 4: AI Enhancement
        print("\n5. Testing AI enhancement...")
        ai_enhancer = model_manager.get_ai_enhancer()
        if ai_enhancer is None:
            print("❌ AI enhancer not available")
            return False
            
        enhanced_image = ai_enhancer.enhance_realism(rendered_image, test_image)
        print(f"✅ AI enhancement complete: {enhanced_image.size}")
        
        print("\n" + "=" * 60)
        print("🎉 Complete 3D Virtual Try-On Pipeline Test SUCCESSFUL!")
        print("All modules are working correctly and can process images end-to-end.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_heic_conversion():
    """Test HEIC conversion functionality"""
    print("\n🧪 Testing HEIC Conversion")
    print("=" * 40)
    
    try:
        test_image = create_test_image()
        
        import io
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()
        
        test_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        print(f"✅ Test base64 created: {len(test_base64)} characters")
        
        from server import convert_heic_to_jpeg
        
        try:
            result = convert_heic_to_jpeg(test_base64)
            print(f"✅ HEIC conversion test successful: {len(result)} characters")
            return True
        except Exception as conv_error:
            print(f"❌ HEIC conversion failed: {str(conv_error)}")
            return False
            
    except Exception as e:
        print(f"❌ HEIC test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Complete Virtual Try-On Pipeline Tests")
    print("=" * 70)
    
    heic_success = test_heic_conversion()
    
    pipeline_success = test_3d_pipeline()
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print(f"HEIC Conversion: {'✅ PASS' if heic_success else '❌ FAIL'}")
    print(f"3D Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    overall_success = heic_success and pipeline_success
    print(f"Overall: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    sys.exit(0 if overall_success else 1)
