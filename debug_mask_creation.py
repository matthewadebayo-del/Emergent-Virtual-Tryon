#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import cv2
from complete_garment_replacement import PracticalGarmentReplacer

def test_mask_creation():
    """Test mask creation with the exact landmarks from production logs"""
    
    # Create test image (512x384 as shown in logs)
    height, width = 512, 384
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Exact landmarks from production logs
    landmarks = {
        'left_shoulder': (243, 195),
        'right_shoulder': (177, 194), 
        'left_hip': (224, 319),
        'right_hip': (189, 318)
    }
    
    print(f"TEST: Image dimensions: {width}x{height}")
    print(f"TEST: Landmarks: {landmarks}")
    
    # Create customer analysis dict
    customer_analysis = {
        'pose_landmarks': {
            'left_shoulder': {'x': 243/width, 'y': 195/height, 'confidence': 1.0},
            'right_shoulder': {'x': 177/width, 'y': 194/height, 'confidence': 1.0},
            'left_hip': {'x': 224/width, 'y': 319/height, 'confidence': 1.0},
            'right_hip': {'x': 189/width, 'y': 318/height, 'confidence': 1.0}
        }
    }
    
    # Test mask creation
    replacer = PracticalGarmentReplacer()
    mask = replacer._create_complete_removal_mask(customer_analysis, (height, width, 3))
    
    mask_area = np.sum(mask > 128)
    coverage = (mask_area / (width * height)) * 100
    
    print(f"TEST RESULT: Mask area: {mask_area} pixels ({coverage:.1f}% coverage)")
    
    if mask_area > 0:
        print("SUCCESS: Mask created successfully")
        # Save mask for inspection
        cv2.imwrite('debug_mask.png', mask)
        print("Saved mask as debug_mask.png")
    else:
        print("FAILURE: Mask is empty")
        
        # Manual polygon test
        print("\nMANUAL POLYGON TEST:")
        
        # Calculate dimensions exactly as in the code
        shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        torso_height = abs(landmarks['left_hip'][1] - landmarks['left_shoulder'][1])
        
        horizontal_expansion = max(35, int(shoulder_width * 0.25))
        vertical_expansion = max(25, int(torso_height * 0.15))
        
        print(f"Shoulder width: {shoulder_width}")
        print(f"Torso height: {torso_height}")
        print(f"Horizontal expansion: {horizontal_expansion}")
        print(f"Vertical expansion: {vertical_expansion}")
        
        # Create polygon points manually
        ls = landmarks['left_shoulder']
        rs = landmarks['right_shoulder']
        lh = landmarks['left_hip']
        rh = landmarks['right_hip']
        
        polygon_points = np.array([
            (ls[0] - horizontal_expansion, ls[1] - vertical_expansion),
            (rs[0] + horizontal_expansion, rs[1] - vertical_expansion),
            (rs[0] + horizontal_expansion, (rs[1] + rh[1]) // 2),
            (rh[0] + horizontal_expansion//2, rh[1] + vertical_expansion),
            (lh[0] - horizontal_expansion//2, lh[1] + vertical_expansion),
            (ls[0] - horizontal_expansion, (ls[1] + lh[1]) // 2),
        ], dtype=np.int32)
        
        print(f"Raw polygon points: {polygon_points}")
        
        # Clip to bounds
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, width - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, height - 1)
        
        print(f"Clipped polygon points: {polygon_points}")
        
        # Test manual mask creation
        manual_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(manual_mask, [polygon_points], 255)
        
        manual_area = np.sum(manual_mask > 128)
        manual_coverage = (manual_area / (width * height)) * 100
        
        print(f"Manual mask area: {manual_area} pixels ({manual_coverage:.1f}% coverage)")
        
        if manual_area > 0:
            cv2.imwrite('debug_manual_mask.png', manual_mask)
            print("Saved manual mask as debug_manual_mask.png")

if __name__ == "__main__":
    test_mask_creation()