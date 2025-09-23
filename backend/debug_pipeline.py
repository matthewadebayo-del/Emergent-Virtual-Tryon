"""
🔍 Backend Debug Checklist - Pipeline Component Testing
"""
import os
import sys
import traceback
from PIL import Image
import io
import base64

def debug_pipeline_components():
    """Debug each component individually"""
    
    print("🔍 DEBUGGING PIPELINE COMPONENTS")
    print("=" * 50)
    
    # Test 1: Check if MediaPipe works
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose()
        print("✅ MediaPipe loaded successfully")
    except Exception as e:
        print(f"❌ MediaPipe failed: {e}")
        traceback.print_exc()
    
    # Test 2: Check if PyBullet works
    try:
        import pybullet as p
        print("✅ PyBullet loaded successfully")
    except Exception as e:
        print(f"❌ PyBullet failed: {e}")
        traceback.print_exc()
    
    # Test 3: Check if Blender works  
    try:
        import subprocess
        result = subprocess.run(['/snap/bin/blender', '--background', '--python-expr', 'import bpy; print("Blender OK")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Blender working")
        else:
            print(f"❌ Blender failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Blender failed: {e}")
        traceback.print_exc()
    
    # Test 4: Check if Stable Diffusion works
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        print("✅ Stable Diffusion imports working")
    except Exception as e:
        print(f"❌ Stable Diffusion failed: {e}")
        traceback.print_exc()
    
    # Test 5: Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ No GPU - will be slower")
    except Exception as e:
        print(f"❌ Torch/GPU check failed: {e}")
    
    # Test 6: Check critical paths
    critical_paths = [
        "/home/mat_a/virtualfit/backend/src/",
        "/home/mat_a/virtualfit/backend/assets/",
        "/tmp/",
    ]
    
    for path in critical_paths:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                print(f"✅ {path}: {len(files)} files")
            except:
                print(f"⚠️ {path}: exists but can't list")
        else:
            print(f"❌ Missing: {path}")

def validate_user_photo(photo_data):
    """Validate input photo format and quality"""
    try:
        if not photo_data or not isinstance(photo_data, str):
            raise Exception("Invalid photo data type")
        
        if not photo_data.startswith('data:image/'):
            raise Exception("Invalid photo format - missing data URL prefix")
        
        # Decode and check dimensions
        image_data = photo_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.size[0] < 200 or image.size[1] < 200:
            raise Exception(f"Photo too small: {image.size}")
        
        print(f"✅ Photo validation passed: {image.size}, mode: {image.mode}")
        return image
        
    except Exception as e:
        print(f"❌ Photo validation failed: {e}")
        raise

def debug_model_manager():
    """Debug the model manager components"""
    try:
        from src.core.model_manager import model_manager
        
        print("🔍 DEBUGGING MODEL MANAGER")
        print("=" * 30)
        
        # Test body reconstructor
        body_reconstructor = model_manager.get_body_reconstructor()
        print(f"Body reconstructor: {'✅ Available' if body_reconstructor else '❌ None'}")
        
        # Test garment fitter
        garment_fitter = model_manager.get_garment_fitter()
        print(f"Garment fitter: {'✅ Available' if garment_fitter else '❌ None'}")
        
        # Test renderer
        renderer = model_manager.get_renderer()
        print(f"Renderer: {'✅ Available' if renderer else '❌ None'}")
        
        # Test AI enhancer
        ai_enhancer = model_manager.get_ai_enhancer()
        print(f"AI enhancer: {'✅ Available' if ai_enhancer else '❌ None'}")
        
    except Exception as e:
        print(f"❌ Model manager debug failed: {e}")
        traceback.print_exc()

def test_basic_image_processing():
    """Test basic image processing without 3D pipeline"""
    try:
        print("🔍 TESTING BASIC IMAGE PROCESSING")
        print("=" * 35)
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='blue')
        
        # Convert to base64
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        test_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        print(f"✅ Test image created: {len(test_base64)} chars")
        
        # Test image operations
        test_image_resized = test_image.resize((256, 256))
        print(f"✅ Image resize: {test_image_resized.size}")
        
        return test_base64
        
    except Exception as e:
        print(f"❌ Basic image processing failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 STARTING BACKEND DEBUG SESSION")
    print("=" * 50)
    
    debug_pipeline_components()
    print("\n")
    
    debug_model_manager()
    print("\n")
    
    test_basic_image_processing()
    print("\n")
    
    print("🏁 DEBUG SESSION COMPLETE")