#!/usr/bin/env python3
"""Test script for enhanced measurement system"""

import sys
import os
import cv2
import numpy as np
import base64
from PIL import Image
import io

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_measurements():
    """Test the enhanced measurement extraction system"""
    
    try:
        from src.core.advanced_measurement_extractor import AdvancedMeasurementExtractor, BodyMeasurements
        print("✅ Successfully imported AdvancedMeasurementExtractor")
    except ImportError as e:
        print(f"❌ Failed to import AdvancedMeasurementExtractor: {e}")
        return False
    
    # Initialize extractor
    try:
        extractor = AdvancedMeasurementExtractor()
        print("✅ AdvancedMeasurementExtractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return False
    
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:400, 200:440] = [255, 255, 255]  # White rectangle representing a person
    
    try:
        reference = {'type': 'height', 'value_cm': 170}
        measurements = extractor.extract_measurements_single_image(test_image, reference)
        
        print("✅ Measurement extraction completed")
        print(f"Height: {measurements.height_cm} cm")
        print(f"Chest circumference: {measurements.chest_circumference} cm")
        print(f"Waist circumference: {measurements.waist_circumference} cm")
        print(f"Hip circumference: {measurements.hip_circumference} cm")
        print(f"Shoulder width: {measurements.shoulder_width} cm")
        print(f"Confidence scores: {len(measurements.confidence_scores)} measurements")
        
        return True
        
    except Exception as e:
        print(f"❌ Measurement extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_body_reconstructor_integration():
    """Test integration with BodyReconstructor"""
    
    try:
        from src.core.body_reconstruction import BodyReconstructor
        print("✅ Successfully imported BodyReconstructor")
    except ImportError as e:
        print(f"❌ Failed to import BodyReconstructor: {e}")
        return False
    
    try:
        reconstructor = BodyReconstructor()
        print("✅ BodyReconstructor initialized successfully")
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:400, 200:440] = [255, 255, 255]
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', test_image)
        image_bytes = buffer.tobytes()
        
        result = reconstructor.process_image_bytes(image_bytes)
        measurements = result["measurements"]
        
        print("✅ Body reconstruction completed")
        print(f"Measurement source: {measurements.get('measurement_source', 'unknown')}")
        print(f"Confidence score: {measurements.get('confidence_score', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Body reconstructor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_garment_measurements():
    """Test garment-specific measurement mappings"""
    
    try:
        from src.core.advanced_measurement_extractor import GARMENT_MEASUREMENTS
        print("✅ Successfully imported GARMENT_MEASUREMENTS")
        
        print(f"Available garment types: {list(GARMENT_MEASUREMENTS.keys())}")
        
        shirt_measurements = GARMENT_MEASUREMENTS['shirts']
        print(f"Shirt primary measurements: {shirt_measurements['primary']}")
        print(f"Shirt secondary measurements: {shirt_measurements['secondary']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Garment measurements test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Enhanced Measurement System ===")
    
    success1 = test_enhanced_measurements()
    success2 = test_body_reconstructor_integration()
    success3 = test_garment_measurements()
    
    if success1 and success2 and success3:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
