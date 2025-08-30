#!/usr/bin/env python3
"""
Test fal.ai Integration with Real API Key
Verify the fal.ai premium pipeline uses real API calls instead of fallback
"""
import sys
import os
sys.path.append('backend')

import asyncio
import base64

async def test_fal_ai_integration():
    """Test the fal.ai premium pipeline with real API key"""
    print("🧪 Testing fal.ai Premium Pipeline with Real API Key")
    print("=" * 60)
    
    try:
        from backend.virtual_tryon_engine import process_virtual_tryon_request
        
        test_image = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
        garment_info = {
            'description': 'Men\'s navy polo shirt',
            'category': 'shirt',
            'image_base64': test_image
        }
        measurements = {'height': 175, 'chest': 90, 'waist': 80}
        
        print("✅ Test data created")
        
        print("\n🔧 Testing fal.ai Pipeline with Real API Key...")
        result = await process_virtual_tryon_request(
            user_image_base64=test_image,
            garment_info=garment_info,
            measurements=measurements,
            method_str='fal_ai',
            openai_api_key='test-key',
            fal_api_key='ed77c72f-02a4-4607-b01d-39cfd6a30ea9:587b4f078dfe4a0a7a694d85ce10042c'
        )
        
        print("✅ fal.ai pipeline test completed!")
        print(f"   Method: {result['technical_details']['method']}")
        print(f"   Stages: {result['technical_details']['pipeline_stages']}")
        print(f"   Size recommendation: {result['size_recommendation']}")
        
        if result['technical_details']['method'] == 'fal_ai_premium':
            print("✅ SUCCESS: Using real fal.ai API services!")
            print(f"   Stage 1 Pose Detected: {result['technical_details'].get('stage1_pose_detected', 'N/A')}")
            print(f"   Stage 2 Model Used: {result['technical_details'].get('stage2_model_used', 'N/A')}")
            print(f"   Stage 3 Enhancement: {result['technical_details'].get('stage3_enhancement_applied', 'N/A')}")
            return True
        else:
            print(f"⚠️  FALLBACK: Fell back to {result['technical_details']['method']}")
            print("   This means fal.ai API key may not be working properly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_environment_variables():
    """Test that environment variables are loaded correctly"""
    print("\n🔧 Testing Environment Variables...")
    
    try:
        import os
        from dotenv import load_dotenv
        
        load_dotenv('/home/ubuntu/repos/Emergent-Virtual-Tryon/backend/.env')
        
        fal_key = os.getenv('FAL_KEY')
        if fal_key:
            print(f"✅ FAL_KEY loaded: {fal_key[:20]}...{fal_key[-10:]}")
            return True
        else:
            print("❌ FAL_KEY not found in environment")
            return False
            
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("🚀 Starting fal.ai Integration Tests")
    print("=" * 80)
    
    env_test = await test_environment_variables()
    api_test = await test_fal_ai_integration()
    
    print("\n" + "=" * 80)
    if env_test and api_test:
        print("🎉 All fal.ai integration tests passed!")
        print("✅ fal.ai API key is properly integrated and working!")
        return 0
    else:
        print("❌ Some integration tests failed!")
        if not env_test:
            print("   - Environment variable loading failed")
        if not api_test:
            print("   - fal.ai API integration failed")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
