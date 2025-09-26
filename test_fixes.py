#!/usr/bin/env python3
"""
Simple test to verify all 4 fixes are working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from comprehensive_tryon import ComprehensiveRegionTryOn, GarmentType
import numpy as np
import cv2

def test_fixes():
    print("Testing Virtual Try-On Fixes...")
    print("=" * 50)
    
    # Create test processor
    processor = ComprehensiveRegionTryOn()
    
    # Test 1: Color Detection Fix
    print("1. Testing Color Detection Fix...")
    garment_analysis = {
        'dominant_colors': [(146, 144, 148)],  # Gray color from analysis
        'texture_features': {'roughness': 0.3}
    }
    product_info = {'name': 'Classic White T-Shirt'}
    
    # Create test image and mask
    test_image = np.ones((400, 300, 3), dtype=np.uint8) * 128
    test_mask = np.ones((400, 300), dtype=np.uint8) * 255
    
    garment_region = processor._generate_garment_appearance(
        GarmentType.TOP, garment_analysis, product_info, test_image.shape, test_mask
    )
    
    # Check if white color is used (should be [255, 255, 255])
    avg_color = np.mean(garment_region, axis=(0, 1))
    if avg_color[0] > 250 and avg_color[1] > 250 and avg_color[2] > 250:
        print("   [OK] Color detection fix working: White garment correctly detected")
    else:
        print(f"   [FAIL] Color detection issue: Got {avg_color}")
    
    # Test 2: Mask Creation Fix
    print("2. Testing Mask Creation Fix...")
    customer_analysis = {
        'pose_landmarks': {
            'left_shoulder': [0.3, 0.2],   # Array format
            'right_shoulder': [0.7, 0.2],
            'left_hip': [0.35, 0.6],
            'right_hip': [0.65, 0.6]
        }
    }
    
    mask = processor._create_top_mask(customer_analysis['pose_landmarks'], (400, 300))
    if mask is not None and np.sum(mask > 128) > 1000:
        print("   [OK] Mask creation fix working: Mask created successfully")
    else:
        print("   [FAIL] Mask creation issue: Failed to create proper mask")
    
    # Test 3: Blending Fix
    print("3. Testing Blending Fix...")
    base_image = np.ones((400, 300, 3), dtype=np.uint8) * 100
    garment_region = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    blended = processor._blend_region_with_image(base_image, garment_region, test_mask, GarmentType.TOP)
    
    # Check if blending occurred (should be between base and garment values)
    avg_blended = np.mean(blended)
    if 120 < avg_blended < 180:  # Should be between 100 and 200
        print("   [OK] Blending fix working: Proper alpha blending applied")
    else:
        print(f"   [FAIL] Blending issue: Average value {avg_blended}")
    
    # Test 4: Quality Assessment Fix
    print("4. Testing Quality Assessment Fix...")
    original = np.ones((400, 300, 3), dtype=np.uint8) * 100
    result = np.ones((400, 300, 3), dtype=np.uint8) * 150  # 50% change
    
    quality_score = processor._calculate_quality_score(result, original, ['top'])
    
    # Should detect significant change and give good score
    if quality_score > 0.5:
        print("   [OK] Quality assessment fix working: Proper change detection")
    else:
        print(f"   [FAIL] Quality assessment issue: Score {quality_score}")
    
    print("=" * 50)
    print("Fix verification complete!")

if __name__ == "__main__":
    test_fixes()