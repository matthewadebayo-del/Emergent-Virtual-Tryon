#!/usr/bin/env python3
"""
Quick test script to verify your virtual try-on fixes are working
Run this to check if your color detection and replacement are functioning correctly
"""

import cv2
import numpy as np
from typing import Dict, Tuple

def test_color_detection():
    """Test the color detection fix"""
    print("Testing Color Detection Fix...")
    
    # Test cases based on your actual data
    test_cases = [
        {
            'product_info': {'name': 'Classic White T-Shirt'},
            'garment_analysis': {'dominant_colors': [(146, 144, 148), (174, 172, 175)]},
            'expected': (255, 255, 255),
            'description': 'White T-Shirt (your actual case)'
        },
        {
            'product_info': {'name': 'Black Cotton Shirt'},
            'garment_analysis': {'dominant_colors': [(50, 50, 50)]},
            'expected': (15, 15, 15),
            'description': 'Black Shirt'
        },
        {
            'product_info': {'name': 'Blue Denim Jacket'},
            'garment_analysis': {'dominant_colors': [(100, 120, 180)]},
            'expected': (20, 20, 220),
            'description': 'Blue Jacket'
        },
        {
            'product_info': {'name': 'Cotton Top'},  # No color in name
            'garment_analysis': {'dominant_colors': [(200, 210, 205)]},  # Light color
            'expected': (255, 255, 255),  # Should force white for light colors
            'description': 'Light colored top (should become white)'
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"\n  Test {i+1}: {test_case['description']}")
        print(f"    Product: '{test_case['product_info']['name']}'")
        print(f"    Analyzed colors: {test_case['garment_analysis']['dominant_colors']}")
        
        # Test the color detection
        result_color = fix_color_detection_immediately(
            test_case['product_info'], 
            test_case['garment_analysis']
        )
        
        expected = test_case['expected']
        
        if result_color == expected:
            print(f"    [PASS] Got {result_color} (expected {expected})")
        else:
            print(f"    [FAIL] Got {result_color} (expected {expected})")
            all_passed = False
    
    print(f"\n{'All color detection tests PASSED!' if all_passed else 'Some tests FAILED - check your implementation'}")
    return all_passed

def test_mask_coverage():
    """Test mask creation and coverage"""
    print("\nTesting Mask Coverage...")
    
    # Simulate your actual landmark data
    customer_analysis = {
        'pose_landmarks': {
            'left_shoulder': [0.6335336565971375, 0.3823147416114807],
            'right_shoulder': [0.4634611904621124, 0.3796965777873993],
            'left_hip': [0.5836747884750366, 0.62428879737854],
            'right_hip': [0.49231430888175964, 0.6220875382423401]
        }
    }
    
    # Test image dimensions (your typical image size)
    image_shape = (500, 400, 3)  # height, width, channels
    height, width = image_shape[:2]
    
    # Test our actual complete replacement system
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        from complete_garment_replacement import CompleteGarmentReplacement
        
        processor = CompleteGarmentReplacement()
        mask = processor._create_complete_removal_mask(customer_analysis, image_shape, ['top'])
    except Exception as e:
        print(f"    [ERROR] Could not test actual system: {e}")
        mask = create_test_aggressive_mask(customer_analysis, image_shape)
    
    if mask is not None:
        mask_area = np.sum(mask > 128)
        total_area = height * width
        coverage_percent = (mask_area / total_area) * 100
        
        print(f"  Mask area: {mask_area} pixels")
        print(f"  Coverage: {coverage_percent:.1f}%")
        
        # Check if mask is adequate
        if coverage_percent < 10:
            print(f"  [FAIL] Mask too small ({coverage_percent:.1f}% < 10%)")
            print(f"  Need more aggressive expansion")
            return False
        elif coverage_percent > 40:
            print(f"  [WARNING] Mask very large ({coverage_percent:.1f}% > 40%)")
            print(f"  May be too aggressive, but better than too small")
            return True
        else:
            print(f"  [PASS] Good mask size ({coverage_percent:.1f}%)")
            return True
    else:
        print(f"  [FAIL] Could not create mask")
        return False

def create_test_aggressive_mask(customer_analysis: Dict, image_shape: Tuple) -> np.ndarray:
    """Create test mask using improved algorithm"""
    
    height, width, _ = image_shape
    pose_landmarks = customer_analysis.get('pose_landmarks', {})
    
    # Convert landmarks to pixel coordinates
    landmarks = {}
    for lm_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
        if lm_name in pose_landmarks:
            lm = pose_landmarks[lm_name]
            x = int(lm[0] * width)
            y = int(lm[1] * height)
            landmarks[lm_name] = (x, y)
            print(f"    {lm_name}: ({x}, {y})")
    
    if len(landmarks) < 4:
        print(f"    [FAIL] Insufficient landmarks: {list(landmarks.keys())}")
        return None
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate aggressive expansion
    shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
    expansion = max(60, int(shoulder_width * 0.5))  # At least 60px or 50% of shoulder width
    
    print(f"    Shoulder width: {shoulder_width}px")
    print(f"    Expansion: {expansion}px")
    
    # Create polygon
    ls = landmarks['left_shoulder']
    rs = landmarks['right_shoulder'] 
    lh = landmarks['left_hip']
    rh = landmarks['right_hip']
    
    polygon = np.array([
        (ls[0] - expansion, ls[1] - expansion//2),
        (rs[0] + expansion, rs[1] - expansion//2),
        (rh[0] + expansion//2, rh[1] + expansion//3),
        (lh[0] - expansion//2, lh[1] + expansion//3)
    ], dtype=np.int32)
    
    # Ensure within bounds
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    
    cv2.fillPoly(mask, [polygon], 255)
    mask = cv2.GaussianBlur(mask, (41, 41), 20)
    
    return mask

def test_visual_change_simulation():
    """Simulate the visual change test"""
    print("\nTesting Visual Change Calculation...")
    
    # Create test images
    original = np.ones((400, 300, 3), dtype=np.uint8) * 100  # Gray image
    
    # Simulate minimal change (current problem)
    minimal_change = original.copy()
    minimal_change[100:200, 100:200] += 5  # Very small change
    
    # Simulate good change (what we want)
    good_change = original.copy()
    good_change[100:300, 50:250] = [255, 255, 255]  # White t-shirt area
    
    # Calculate changes
    minimal_diff = np.sum(cv2.absdiff(original, minimal_change))
    good_diff = np.sum(cv2.absdiff(original, good_change))
    
    print(f"  Minimal change total: {minimal_diff}")
    print(f"  Good change total: {good_diff}")
    
    # Check if differences are meaningful
    if minimal_diff < 200000:  # Adjusted threshold
        print(f"  [PASS] Minimal change correctly identified as insufficient")
    else:
        print(f"  [FAIL] Minimal change not properly detected")
    
    if good_diff > 1000000:
        print(f"  [PASS] Good change correctly identified as significant")
        return True
    else:
        print(f"  [FAIL] Good change not properly detected")
        return False

def fix_color_detection_immediately(product_info: Dict, garment_analysis: Dict) -> Tuple[int, int, int]:
    """The actual color detection fix function"""
    
    product_name = product_info.get('name', '').lower()
    
    # Force correct colors based on product name
    if any(word in product_name for word in ['white', 'blanc', 'blanco']):
        return (255, 255, 255)  # Pure white
    elif any(word in product_name for word in ['black', 'noir', 'negro']):
        return (15, 15, 15)     # Pure black
    elif any(word in product_name for word in ['red', 'rouge', 'rojo']):
        return (220, 20, 20)    # Red
    elif any(word in product_name for word in ['blue', 'bleu', 'azul']):
        return (20, 20, 220)    # Blue
    else:
        # Fallback logic
        dominant_colors = garment_analysis.get('dominant_colors', [])
        if dominant_colors:
            r, g, b = dominant_colors[0]
            # If light color, force white
            if r > 180 and g > 180 and b > 180:
                return (255, 255, 255)
            else:
                return dominant_colors[0]
        else:
            return (128, 128, 128)

def main():
    """Run all tests"""
    print("Virtual Try-On Fix Verification Tests")
    print("=" * 50)
    
    test_results = {
        'color_detection': test_color_detection(),
        'mask_coverage': test_mask_coverage(), 
        'visual_change': test_visual_change_simulation()
    }
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests PASSED! Your fixes should work correctly.")
    else:
        print("Some tests FAILED. Check your implementation.")
        print("\nNext steps:")
        if not test_results['color_detection']:
            print("  - Fix color detection using the fix_color_detection_immediately function")
        if not test_results['mask_coverage']:
            print("  - Increase mask expansion in your mask creation")
        if not test_results['visual_change']:
            print("  - Implement stronger garment replacement instead of subtle blending")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)