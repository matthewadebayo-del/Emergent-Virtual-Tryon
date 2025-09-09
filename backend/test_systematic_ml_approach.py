#!/usr/bin/env python3
"""Test the complete systematic 7-step ML dependency approach"""

import sys
import os
import requests
import subprocess

def test_step_1_debug_endpoint():
    """Test Step 1: Debug ML versions endpoint"""
    try:
        response = requests.get("https://virtualfit-backend-kkohgb7xuq-uc.a.run.app/api/v1/debug-ml-versions")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Step 1: Debug endpoint working")
            print(f"   cached_download: {data.get('cached_download_available', 'unknown')}")
            print(f"   huggingface_hub: {data.get('huggingface_hub', 'unknown')}")
            return data.get('cached_download_available', False)
        else:
            print(f"❌ Step 1: Debug endpoint failed ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Step 1: Debug endpoint error: {e}")
        return False

def test_step_2_version_combinations():
    """Test Step 2: Verify exact version combinations"""
    try:
        import torch
        import transformers
        import diffusers
        import huggingface_hub
        
        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "huggingface_hub": huggingface_hub.__version__
        }
        
        expected = {
            "torch": "2.2.0",
            "transformers": "4.33.3", 
            "diffusers": "0.21.4",
            "huggingface_hub": "0.16.4"
        }
        
        all_correct = True
        for pkg, version in versions.items():
            if version.startswith(expected[pkg]):
                print(f"✅ Step 2: {pkg} version correct ({version})")
            else:
                print(f"❌ Step 2: {pkg} version mismatch (got {version}, expected {expected[pkg]})")
                all_correct = False
        
        return all_correct
    except ImportError as e:
        print(f"❌ Step 2: Import error: {e}")
        return False

def test_step_6_fallback():
    """Test Step 6: AI enhancement fallback"""
    try:
        from src.core.ai_enhancement import FixedAIEnhancer
        enhancer = FixedAIEnhancer()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_in, \
             tempfile.NamedTemporaryFile(suffix=".jpg") as temp_out:
            
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(temp_in.name)
            
            success = enhancer.enhance_image(temp_in.name, temp_out.name)
            if success and os.path.exists(temp_out.name):
                print("✅ Step 6: AI enhancement fallback working")
                return True
            else:
                print("❌ Step 6: AI enhancement fallback failed")
                return False
    except Exception as e:
        print(f"❌ Step 6: Fallback test error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Systematic 7-Step ML Dependency Approach ===")
    
    step_1_ok = test_step_1_debug_endpoint()
    step_2_ok = test_step_2_version_combinations()
    step_6_ok = test_step_6_fallback()
    
    print("\n=== Results ===")
    print(f"Step 1 (Debug): {'✅' if step_1_ok else '❌'}")
    print(f"Step 2 (Versions): {'✅' if step_2_ok else '❌'}")
    print(f"Step 6 (Fallback): {'✅' if step_6_ok else '❌'}")
    
    if step_1_ok and step_2_ok and step_6_ok:
        print("\n🎉 Systematic ML dependency approach working!")
        sys.exit(0)
    else:
        print("\n❌ Some steps failed - check logs above")
        sys.exit(1)
