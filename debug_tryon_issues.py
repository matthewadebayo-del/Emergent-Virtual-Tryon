#!/usr/bin/env python3
"""
Debug script to identify and fix virtual try-on issues
Run this to diagnose what's going wrong with your virtual try-on system
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_virtual_tryon_system(customer_analysis: Dict, garment_analysis: Dict, 
                             product_info: Dict, original_image: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive debugging function to identify virtual try-on issues
    """
    logger.info("🔍 Starting comprehensive virtual try-on debugging...")
    
    debug_results = {
        'issues_found': [],
        'recommendations': [],
        'data_analysis': {},
        'image_analysis': {},
        'processing_test': {}
    }
    
    # 1. ANALYZE INPUT DATA
    logger.info("📊 Analyzing input data...")
    
    # Check customer analysis
    pose_landmarks = customer_analysis.get('pose_landmarks', {})
    required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    
    debug_results['data_analysis']['landmarks_found'] = list(pose_landmarks.keys())
    debug_results['data_analysis']['required_landmarks'] = required_landmarks
    missing_landmarks = [lm for lm in required_landmarks if lm not in pose_landmarks]
    
    if missing_landmarks:
        debug_results['issues_found'].append(f"❌ Missing landmarks: {missing_landmarks}")
        debug_results['recommendations'].append("Check pose detection - ensure full body is visible")
    else:
        logger.info("✅ All required landmarks present")
    
    # Check landmark confidence
    low_confidence_landmarks = []
    for landmark in required_landmarks:
        if landmark in pose_landmarks:
            lm_data = pose_landmarks[landmark]
            if isinstance(lm_data, dict):
                confidence = lm_data.get('confidence', 1.0)
            else:
                confidence = 1.0  # Default for array format
            if confidence < 0.7:
                low_confidence_landmarks.append(f"{landmark}({confidence:.2f})")
    
    if low_confidence_landmarks:
        debug_results['issues_found'].append(f"⚠️ Low confidence landmarks: {low_confidence_landmarks}")
        debug_results['recommendations'].append("Improve lighting and pose clarity")
    
    # 2. ANALYZE GARMENT DATA
    logger.info("👕 Analyzing garment data...")
    
    dominant_colors = garment_analysis.get('dominant_colors', [])
    product_name = product_info.get('name', '').lower()
    
    debug_results['data_analysis']['dominant_colors'] = dominant_colors
    debug_results['data_analysis']['product_name'] = product_name
    debug_results['data_analysis']['name_contains_white'] = 'white' in product_name
    
    # Check color analysis issue
    if 'white' in product_name:
        if not dominant_colors:
            debug_results['issues_found'].append("❌ No dominant colors found for white garment")
        else:
            primary_color = dominant_colors[0]
            # Check if primary color is actually white-ish
            r, g, b = primary_color
            if not (r > 200 and g > 200 and b > 200):
                debug_results['issues_found'].append(
                    f"❌ WHITE garment detected with NON-WHITE primary color: {primary_color}")
                debug_results['recommendations'].append(
                    "Fix color analysis: white garments should have white primary colors")
    
    # 3. ANALYZE IMAGE
    logger.info("🖼️ Analyzing original image...")
    
    if original_image is None or original_image.size == 0:
        debug_results['issues_found'].append("❌ Invalid original image")
        return debug_results
    
    debug_results['image_analysis'] = {
        'shape': original_image.shape,
        'size': original_image.size,
        'dtype': str(original_image.dtype),
        'min_val': int(np.min(original_image)),
        'max_val': int(np.max(original_image)),
        'mean_brightness': float(np.mean(original_image))
    }
    
    # 4. TEST MASK CREATION
    logger.info("🎭 Testing mask creation...")
    
    try:
        # Simulate mask creation for top garment
        height, width = original_image.shape[:2]
        test_mask = create_test_top_mask(pose_landmarks, (height, width))
        
        if test_mask is not None:
            mask_area = np.sum(test_mask > 128)
            total_area = height * width
            coverage_percent = (mask_area / total_area) * 100
            
            debug_results['processing_test']['mask_created'] = True
            debug_results['processing_test']['mask_area'] = int(mask_area)
            debug_results['processing_test']['coverage_percent'] = round(coverage_percent, 2)
            
            if coverage_percent < 5:
                debug_results['issues_found'].append(f"❌ Mask too small: {coverage_percent:.1f}% coverage")
                debug_results['recommendations'].append("Increase mask expansion or check landmark positions")
            elif coverage_percent > 50:
                debug_results['issues_found'].append(f"⚠️ Mask very large: {coverage_percent:.1f}% coverage")
                debug_results['recommendations'].append("Check mask creation logic - may be too aggressive")
            else:
                logger.info(f"✅ Mask size appropriate: {coverage_percent:.1f}% coverage")
                
            # Save mask for visual inspection
            cv2.imwrite('/tmp/debug_mask_visualization.png', test_mask)
            logger.info("💾 Debug mask saved to /tmp/debug_mask_visualization.png")
            
        else:
            debug_results['issues_found'].append("❌ Failed to create mask")
            debug_results['processing_test']['mask_created'] = False
            
    except Exception as e:
        debug_results['issues_found'].append(f"❌ Mask creation error: {str(e)}")
        debug_results['processing_test']['mask_error'] = str(e)
    
    # 5. TEST GARMENT GENERATION
    logger.info("🎨 Testing garment generation...")
    
    try:
        test_garment = create_test_garment(dominant_colors, product_name, original_image.shape)
        
        if test_garment is not None:
            garment_color = test_garment[0, 0]  # Get color from first pixel
            debug_results['processing_test']['garment_generated'] = True
            debug_results['processing_test']['generated_color'] = [int(c) for c in garment_color]
            
            # Check if color matches expectation
            if 'white' in product_name:
                if not (garment_color[0] > 240 and garment_color[1] > 240 and garment_color[2] > 240):
                    debug_results['issues_found'].append(
                        f"❌ White garment generated non-white color: {garment_color}")
            
            logger.info(f"✅ Garment generated with color: {garment_color}")
        else:
            debug_results['issues_found'].append("❌ Failed to generate garment")
            
    except Exception as e:
        debug_results['issues_found'].append(f"❌ Garment generation error: {str(e)}")
    
    # 6. GENERATE RECOMMENDATIONS
    if not debug_results['issues_found']:
        debug_results['recommendations'].append("✅ No major issues found - check blending logic")
    else:
        debug_results['recommendations'].append("🔧 Apply the fixes provided in the debugging artifacts")
    
    # 7. OUTPUT SUMMARY
    logger.info("📋 DEBUGGING SUMMARY:")
    logger.info(f"   Issues found: {len(debug_results['issues_found'])}")
    
    for issue in debug_results['issues_found']:
        logger.warning(f"   {issue}")
    
    for rec in debug_results['recommendations']:
        logger.info(f"   💡 {rec}")
    
    return debug_results

def create_test_top_mask(pose_landmarks: Dict, image_size: tuple) -> np.ndarray:
    """Test mask creation with enhanced debugging"""
    
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get landmarks
    points = {}
    for landmark in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
        if landmark in pose_landmarks:
            lm_data = pose_landmarks[landmark]
            if isinstance(lm_data, (list, tuple)) and len(lm_data) >= 2:
                x = int(lm_data[0] * width)
                y = int(lm_data[1] * height)
                points[landmark] = (x, y)
                logger.info(f"   Landmark {landmark}: ({x}, {y})")
    
    if len(points) < 4:
        logger.error(f"❌ Insufficient landmarks for mask: {list(points.keys())}")
        return None
    
    # Create polygon with expansion
    left_shoulder = points['left_shoulder']
    right_shoulder = points['right_shoulder']
    left_hip = points['left_hip']
    right_hip = points['right_hip']
    
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    expansion = max(30, int(shoulder_width * 0.2))  # More aggressive expansion
    
    polygon_points = [
        (left_shoulder[0] - expansion, left_shoulder[1] - expansion//2),
        (right_shoulder[0] + expansion, right_shoulder[1] - expansion//2),
        (right_hip[0] + expansion, right_hip[1] + expansion//2),
        (left_hip[0] - expansion, left_hip[1] + expansion//2)
    ]
    
    # Ensure points are within bounds
    polygon_points = [(max(0, min(width-1, x)), max(0, min(height-1, y))) for x, y in polygon_points]
    polygon_points = np.array(polygon_points, dtype=np.int32)
    
    cv2.fillPoly(mask, [polygon_points], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    
    return mask

def create_test_garment(dominant_colors: list, product_name: str, image_shape: tuple) -> np.ndarray:
    """Test garment generation with proper color logic"""
    
    height, width, _ = image_shape
    garment = np.zeros((height, width, 3), dtype=np.uint8)
    
    # FIXED color logic
    if 'white' in product_name:
        base_color = (255, 255, 255)
        logger.info(f"   Using WHITE for product: {product_name}")
    elif 'black' in product_name:
        base_color = (0, 0, 0)
        logger.info(f"   Using BLACK for product: {product_name}")
    else:
        base_color = dominant_colors[0] if dominant_colors else (128, 128, 128)
        logger.info(f"   Using analyzed color: {base_color}")
    
    garment[:] = base_color
    return garment

# MAIN EXECUTION FUNCTION
def run_debug_on_current_issue():
    """Run debug on the specific issue from the log"""
    
    logger.info("🚀 Running debug on current virtual try-on issue...")
    
    # Reconstruct the data from your log
    customer_analysis = {
        'pose_landmarks': {
            'nose': [0.5401559472084045, 0.26972973346710205],
            'left_shoulder': [0.6335336565971375, 0.3823147416114807],
            'right_shoulder': [0.4634611904621124, 0.3796965777873993],
            'left_hip': [0.5836747884750366, 0.62428879737854],
            'right_hip': [0.49231430888175964, 0.6220875382423401],
            'left_wrist': [0.7, 0.5],  # Estimated
            'right_wrist': [0.3, 0.5],  # Estimated
            'left_ankle': [0.6, 0.9],  # Estimated
            'right_ankle': [0.4, 0.9]   # Estimated
        }
    }
    
    garment_analysis = {
        'dominant_colors': [(146, 144, 148), (174, 172, 175), (77, 54, 44), (204, 204, 210), (111, 93, 83)],
        'texture_features': {
            'roughness': 0.29114076791771837,
            'edge_density': 0.02881622314453125,
            'complexity': 0.10797401991485317,
            'smooth_factor': 0.7088592320822816
        }
    }
    
    product_info = {
        'name': 'Classic White T-Shirt'
    }
    
    # Create dummy image (replace with actual image in real usage)
    original_image = np.ones((500, 400, 3), dtype=np.uint8) * 128  # Gray dummy image
    
    # Convert landmarks to dict format for testing
    for landmark, coords in customer_analysis['pose_landmarks'].items():
        if isinstance(coords, list):
            customer_analysis['pose_landmarks'][landmark] = {
                'x': coords[0],
                'y': coords[1],
                'confidence': 1.0
            }
    
    # Run debug
    results = debug_virtual_tryon_system(customer_analysis, garment_analysis, product_info, original_image)
    
    # Print results
    print("\n" + "="*50)
    print("VIRTUAL TRY-ON DEBUG RESULTS")
    print("="*50)
    print(f"Issues Found: {len(results['issues_found'])}")
    for issue in results['issues_found']:
        print(f"  {issue}")
    
    print(f"\nRecommendations: {len(results['recommendations'])}")
    for rec in results['recommendations']:
        print(f"  {rec}")
    
    print(f"\nDebug Data: {json.dumps(results, indent=2, default=str)}")
    
    return results

if __name__ == "__main__":
    run_debug_on_current_issue()