#!/usr/bin/env python3
"""
Test script for rendering pipeline fixes
Tests the fixed DirectionalLight API and updated logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.rendering import RenderingPipeline
from src.core.ai_enhancement import AIEnhancer
import trimesh
import numpy as np
from PIL import Image

def test_rendering_pipeline():
    """Test the fixed rendering pipeline"""
    print("Testing rendering pipeline fixes...")
    
    body_mesh = trimesh.creation.cylinder(radius=0.3, height=1.8)
    garment_mesh = trimesh.creation.cylinder(radius=0.32, height=0.6)
    
    renderer = RenderingPipeline()
    
    try:
        result = renderer.render_scene(
            body_mesh=body_mesh,
            garment_mesh=garment_mesh,
            output_path="/tmp/test_render.png",
            fabric_type="cotton",
            fabric_color=(0.2, 0.3, 0.8)
        )
        print(f"✅ Rendering pipeline test passed: {result}")
        return True
    except Exception as e:
        print(f"❌ Rendering pipeline test failed: {e}")
        return False

def test_ai_enhancement():
    """Test the fixed AI enhancement"""
    print("Testing AI enhancement fixes...")
    
    enhancer = AIEnhancer()
    
    try:
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_input_path = "/tmp/test_input.png"
        test_output_path = "/tmp/test_output.png"
        
        Image.fromarray(test_image.astype('uint8')).save(test_input_path)
        
        result = enhancer.enhance_image(test_input_path, test_output_path)
        
        if result and os.path.exists(test_output_path):
            print(f"✅ AI enhancement test passed")
            return True
        else:
            print(f"❌ AI enhancement test failed - no output file")
            return False
    except Exception as e:
        print(f"❌ AI enhancement test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running rendering fixes tests...")
    
    test1_passed = test_rendering_pipeline()
    test2_passed = test_ai_enhancement()
    
    if test1_passed and test2_passed:
        print("✅ All rendering fixes tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
