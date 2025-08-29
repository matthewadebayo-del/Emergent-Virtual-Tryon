#!/usr/bin/env python3
"""
Test fal.ai Premium 3-Stage Pipeline
"""
import sys
import os
sys.path.append('backend')

import asyncio
import base64
from PIL import Image
import io

def create_test_image(width=512, height=768, color=(200, 220, 240)):
    """Create a test image"""
    image = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

async def test_fal_ai_premium_pipeline():
    """Test the fal.ai premium 3-stage pipeline"""
    print("🧪 Testing fal.ai Premium 3-Stage Pipeline")
    print("=" * 60)
    
    try:
        from backend.fal_ai_premium_pipeline import FalAIPremiumPipeline
        
        pipeline = FalAIPremiumPipeline(fal_api_key=None)
        print("✅ FalAIPremiumPipeline initialized")
        
        user_image_bytes = create_test_image(512, 768, (220, 200, 180))
        garment_image_bytes = create_test_image(256, 256, (100, 150, 200))
        
        measurements = {
            'height': 175,
            'chest': 95,
            'waist': 80,
            'hips': 100
        }
        
        print("✅ Test data created")
        
        result = await pipeline.process_virtual_tryon(
            user_image_bytes=user_image_bytes,
            garment_image_bytes=garment_image_bytes,
            garment_description="Men's blue polo shirt",
            measurements=measurements
        )
        
        print("✅ Pipeline processing completed")
        
        assert 'result_image_base64' in result
        assert 'size_recommendation' in result
        assert 'technical_details' in result
        
        technical_details = result['technical_details']
        assert technical_details['method'] == 'fal_ai_premium'
        assert technical_details['pipeline_stages'] == 3
        assert len(technical_details['stages']) == 3
        
        print(f"✅ Size recommendation: {result['size_recommendation']}")
        print(f"✅ Pipeline stages: {technical_details['pipeline_stages']}")
        print(f"✅ Stages: {technical_details['stages']}")
        print(f"✅ Pose detected: {technical_details.get('stage1_pose_detected', 'N/A')}")
        print(f"✅ Model used: {technical_details.get('stage2_model_used', 'N/A')}")
        
        pipeline.cleanup()
        
        print("\n🎉 fal.ai Premium Pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_virtual_tryon_engine_integration():
    """Test integration with virtual try-on engine"""
    print("\n🧪 Testing Virtual Try-On Engine Integration")
    print("=" * 60)
    
    try:
        from backend.virtual_tryon_engine import process_virtual_tryon_request
        
        user_image_bytes = create_test_image(512, 768, (220, 200, 180))
        user_image_base64 = base64.b64encode(user_image_bytes).decode('utf-8')
        
        garment_info = {
            'description': 'Men\'s navy polo shirt',
            'category': 'shirt',
            'image_base64': base64.b64encode(create_test_image(256, 256, (50, 50, 100))).decode('utf-8')
        }
        
        measurements = {'height': 175, 'chest': 95, 'waist': 80}
        
        result = await process_virtual_tryon_request(
            user_image_base64=user_image_base64,
            garment_info=garment_info,
            measurements=measurements,
            method_str="fal_ai",
            openai_api_key="test-key",
            fal_api_key=None
        )
        
        print("✅ Engine integration test completed")
        print(f"✅ Method used: {result['technical_details']['method']}")
        print(f"✅ Personalization note: {result['personalization_note'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting fal.ai Premium Pipeline Tests")
    print("=" * 80)
    
    test1_passed = await test_fal_ai_premium_pipeline()
    test2_passed = await test_virtual_tryon_engine_integration()
    
    print("\n" + "=" * 80)
    if test1_passed and test2_passed:
        print("🎉 All fal.ai Premium Pipeline tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
