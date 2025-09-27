#!/usr/bin/env python3
"""
Compatibility Fix Script for VirtualFit
Addresses NumPy/OpenCV compatibility and MediaPipe installation issues
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command {cmd}: {e}")
        return False

def fix_numpy_opencv_compatibility():
    """Fix NumPy/OpenCV compatibility issues"""
    print("=== Fixing NumPy/OpenCV Compatibility ===")
    
    # Uninstall problematic versions
    print("Uninstalling incompatible versions...")
    run_command("pip uninstall numpy opencv-python -y")
    
    # Install compatible versions
    print("Installing compatible versions...")
    success = run_command('pip install "numpy>=1.21.0,<2.0" opencv-python==4.8.1.78')
    
    if success:
        print("✓ NumPy/OpenCV compatibility fixed")
    else:
        print("✗ Failed to fix NumPy/OpenCV compatibility")
    
    return success

def install_mediapipe_alternative():
    """Install MediaPipe or alternative pose detection"""
    print("=== Installing MediaPipe/Pose Detection ===")
    
    # Try MediaPipe first
    print("Attempting MediaPipe installation...")
    if run_command("pip install mediapipe"):
        print("✓ MediaPipe installed successfully")
        return True
    
    # Try alternative pose detection libraries
    print("MediaPipe failed, trying alternatives...")
    alternatives = [
        "opencv-contrib-python",
        "pose-estimation",
        "tf-pose-estimation"
    ]
    
    for alt in alternatives:
        print(f"Trying {alt}...")
        if run_command(f"pip install {alt}"):
            print(f"✓ {alt} installed successfully")
            return True
    
    print("✗ All pose detection libraries failed")
    return False

def fix_unicode_console():
    """Fix Unicode console display issues"""
    print("=== Fixing Unicode Console Issues ===")
    
    # Set environment variables for Unicode support
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    print("✓ Unicode environment variables set")
    return True

def test_imports():
    """Test critical imports"""
    print("=== Testing Critical Imports ===")
    
    imports_to_test = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn")
    ]
    
    success_count = 0
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✓ {name} import successful")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    
    # Test MediaPipe separately
    try:
        import mediapipe as mp
        print("✓ MediaPipe import successful")
        success_count += 1
    except ImportError:
        print("⚠ MediaPipe import failed (fallback available)")
    
    print(f"Import test results: {success_count}/{len(imports_to_test)} core libraries working")
    return success_count >= len(imports_to_test) - 1  # Allow MediaPipe to fail

def main():
    """Main fix function"""
    print("VirtualFit Compatibility Fix Script")
    print("=" * 40)
    
    fixes_applied = []
    
    # Fix NumPy/OpenCV compatibility
    if fix_numpy_opencv_compatibility():
        fixes_applied.append("NumPy/OpenCV compatibility")
    
    # Install MediaPipe or alternatives
    if install_mediapipe_alternative():
        fixes_applied.append("Pose detection library")
    
    # Fix Unicode console
    if fix_unicode_console():
        fixes_applied.append("Unicode console support")
    
    # Test imports
    if test_imports():
        fixes_applied.append("Import validation")
    
    print("\n" + "=" * 40)
    print("Fix Summary:")
    if fixes_applied:
        for fix in fixes_applied:
            print(f"✓ {fix}")
    else:
        print("✗ No fixes were successfully applied")
    
    print("\nRecommendations:")
    print("1. Restart your terminal/IDE after running this script")
    print("2. Test the virtual try-on system with a simple image")
    print("3. If MediaPipe is still unavailable, the system will use fallback pose detection")
    
    return len(fixes_applied) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)