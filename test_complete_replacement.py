#!/usr/bin/env python3
"""
Test the complete garment replacement system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from complete_garment_replacement import process_complete_garment_replacement
import numpy as np
import cv2

def test_complete_replacement():
    print("Testing Complete Garment Replacement System...")
    print("=" * 50)
    
    # Create test data
    customer_analysis = {
        'pose_landmarks': {
            'left_shoulder': [0.3, 0.2],
            'right_shoulder': [0.7, 0.2],
            'left_hip': [0.35, 0.6],
            'right_hip': [0.65, 0.6],
            'left_elbow': [0.25, 0.35],
            'right_elbow': [0.75, 0.35]
        }
    }
    
    garment_analysis = {
        'dominant_colors': [(146, 144, 148)],
        'texture_features': {'roughness': 0.3}
    }
    
    product_info = {'name': 'Classic White T-Shirt'}
    
    # Create test image
    test_image = np.ones((400, 300, 3), dtype=np.uint8) * 128
    
    # Test complete replacement
    try:
        result_image = process_complete_garment_replacement(
            customer_analysis=customer_analysis,
            garment_analysis=garment_analysis,
            product_info=product_info,
            original_image=test_image,
            garment_types=['top']
        )
        
        if result_image is not None:
            print("[SUCCESS] Complete garment replacement successful!")
            print(f"   Result shape: {result_image.shape}")
            
            # Check if white color was applied
            avg_color = np.mean(result_image, axis=(0, 1))
            print(f"   Average result color: {avg_color}")
            
            # Check for visual change
            diff = cv2.absdiff(result_image, test_image)
            total_change = np.sum(diff)
            print(f"   Total visual change: {total_change}")
            
            if total_change > 10000:
                print("[SUCCESS] Significant visual transformation detected!")
            else:
                print("⚠️ Limited visual change")
                
        else:
            print("[FAILED] Complete garment replacement failed")
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    
    print("=" * 50)
    print("Complete replacement test finished!")

if __name__ == "__main__":
    test_complete_replacement()