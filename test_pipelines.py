#!/usr/bin/env python3
"""
Test script for virtual try-on pipelines without database dependencies
"""

import sys
import base64
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "backend"))

async def test_hybrid_3d_pipeline():
    """Test the Hybrid 3D pipeline directly"""
    print("🔍 Testing Hybrid 3D Pipeline...")
    
    try:
        from virtual_tryon_engine import VirtualTryOnEngine, TryOnMethod
        
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        garment_info = {
            'description': 'Blue cotton t-shirt',
            'category': 'shirt',
            'image_base64': test_image_base64
        }
        
        measurements = {
            'height': 175.0,
            'chest': 95.0,
            'waist': 80.0,
            'hips': 100.0
        }
        
        engine = VirtualTryOnEngine(
            openai_api_key="test_key",
            fal_api_key=None
        )
        
        result = await engine.process_virtual_tryon(
            user_image_base64=test_image_base64,
            garment_info=garment_info,
            measurements=measurements,
            method=TryOnMethod.HYBRID_3D,
            fallback_enabled=True
        )
        
        print("✅ Hybrid 3D pipeline executed successfully")
        print(f"   Method: {result.get('technical_details', {}).get('method', 'unknown')}")
        print(f"   Stages: {result.get('technical_details', {}).get('pipeline_stages', 0)}")
        print(f"   Size recommendation: {result.get('size_recommendation', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid 3D pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_fal_ai_pipeline():
    """Test the fal.ai pipeline (should fallback to Hybrid 3D)"""
    print("\n🔍 Testing fal.ai Pipeline (fallback to Hybrid 3D)...")
    
    try:
        from virtual_tryon_engine import VirtualTryOnEngine, TryOnMethod
        
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        garment_info = {
            'description': 'Red dress',
            'category': 'dress',
            'image_base64': test_image_base64
        }
        
        measurements = {
            'height': 165.0,
            'chest': 85.0,
            'waist': 70.0,
            'hips': 95.0
        }
        
        engine = VirtualTryOnEngine(
            openai_api_key="test_key",
            fal_api_key=None
        )
        
        result = await engine.process_virtual_tryon(
            user_image_base64=test_image_base64,
            garment_info=garment_info,
            measurements=measurements,
            method=TryOnMethod.FAL_AI,
            fallback_enabled=True
        )
        
        print("✅ fal.ai pipeline executed successfully (with fallback)")
        print(f"   Method: {result.get('technical_details', {}).get('method', 'unknown')}")
        print(f"   Stages: {result.get('technical_details', {}).get('pipeline_stages', 0)}")
        print(f"   Size recommendation: {result.get('size_recommendation', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ fal.ai pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_openai_fallback():
    """Test the OpenAI fallback pipeline"""
    print("\n🔍 Testing OpenAI Fallback Pipeline...")
    
    try:
        from virtual_tryon_engine import VirtualTryOnEngine, TryOnMethod
        
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        garment_info = {
            'description': 'Green jacket',
            'category': 'jacket'
        }
        
        measurements = {
            'height': 180.0,
            'chest': 105.0,
            'waist': 90.0
        }
        
        engine = VirtualTryOnEngine(
            openai_api_key="test_key",
            fal_api_key=None
        )
        
        result = await engine.process_virtual_tryon(
            user_image_base64=test_image_base64,
            garment_info=garment_info,
            measurements=measurements,
            method=TryOnMethod.OPENAI_FALLBACK,
            fallback_enabled=False
        )
        
        print("✅ OpenAI fallback pipeline executed successfully")
        print(f"   Method: {result.get('technical_details', {}).get('method', 'unknown')}")
        print(f"   Size recommendation: {result.get('size_recommendation', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI fallback pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all pipeline tests"""
    print("🚀 Testing Virtual Try-On Pipelines")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if await test_hybrid_3d_pipeline():
        tests_passed += 1
    
    if await test_fal_ai_pipeline():
        tests_passed += 1
    
    if await test_openai_fallback():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Pipeline Tests: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All pipeline tests passed!")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} pipeline tests failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
