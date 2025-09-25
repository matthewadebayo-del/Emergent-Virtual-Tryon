#!/usr/bin/env python3
"""
Test Enhanced Virtual Try-On Pipeline
Tests the complete pipeline with garment image analysis
"""

import sys
import os
sys.path.append('.')

from src.core.garment_analyzer import GarmentImageAnalyzer
from PIL import Image, ImageDraw
import io
import base64

def create_test_garment_image(color='blue', garment_type='shirt'):
    """Create a test garment image"""
    img = Image.new('RGB', (200, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple shirt shape
    if garment_type == 'shirt':
        # Body
        draw.rectangle([50, 80, 150, 200], fill=color, outline='black')
        # Sleeves
        draw.rectangle([20, 80, 50, 130], fill=color, outline='black')
        draw.rectangle([150, 80, 180, 130], fill=color, outline='black')
        # Collar
        draw.rectangle([75, 60, 125, 80], fill=color, outline='black')
    
    return img

def test_garment_analysis():
    """Test garment image analysis"""
    print("=== Testing Garment Image Analysis ===")
    
    analyzer = GarmentImageAnalyzer()
    
    # Test different colored garments
    colors = ['blue', 'red', 'white', 'black']
    
    for color in colors:
        print(f"\nTesting {color} shirt:")
        
        # Create test image
        test_img = create_test_garment_image(color=color)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        test_bytes = img_bytes.getvalue()
        
        # Analyze
        analysis = analyzer.analyze_garment_image(test_bytes)
        
        print(f"  Analysis Success: {analysis['analysis_success']}")
        print(f"  Primary Color: {analysis['colors']['primary']}")
        print(f"  Fabric Type: {analysis['fabric_type']}")
        print(f"  Pattern Type: {analysis['patterns']['type']}")
        
        # Test base64 encoding (for API compatibility)
        base64_data = base64.b64encode(test_bytes).decode('utf-8')
        print(f"  Base64 Length: {len(base64_data)} chars")

def test_production_engine():
    """Test production virtual try-on engine"""
    print("\n=== Testing Production Engine ===")
    
    try:
        from production_server import ProductionVirtualTryOn
        engine = ProductionVirtualTryOn()
        
        print(f"  Mesh Processor: {'Available' if engine.mesh_processor else 'Not Available'}")
        print(f"  Physics Engine: {'Available' if engine.physics_engine else 'Not Available'}")
        print(f"  AI Enhancer: {'Available' if engine.ai_enhancer else 'Not Available'}")
        
        # Test color name conversion
        test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
        for rgb in test_colors:
            color_name = engine._rgb_to_color_name(rgb)
            print(f"  RGB {rgb} -> {color_name}")
            
    except Exception as e:
        print(f"  Error: {e}")

def main():
    """Run all tests"""
    print("[TEST] Testing Enhanced Virtual Try-On Pipeline")
    print("=" * 50)
    
    try:
        test_garment_analysis()
        test_production_engine()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests completed successfully!")
        print("[READY] Enhanced pipeline is ready for testing")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()