#!/usr/bin/env python3
"""
Test script to analyze actual dependency usage in the 3D virtual try-on system
"""
import sys
import os
sys.path.append('src')

def test_minimal_requirements():
    """Test minimal requirements functionality"""
    print('Testing minimal requirements functionality...')
    try:
        import numpy as np
        import cv2
        print('✅ Core dependencies (numpy, cv2) available')
    except Exception as e:
        print(f'❌ Core dependencies failed: {e}')

def test_3d_dependencies():
    """Test 3D dependencies usage"""
    print('\nTesting 3D dependencies usage...')
    
    try:
        from src.core.body_reconstruction import BodyReconstructor
        br = BodyReconstructor()
        has_mediapipe = hasattr(br, 'mp_pose') and br.mp_pose is not None
        print(f'✅ BodyReconstructor initialized (MediaPipe available: {has_mediapipe})')
    except Exception as e:
        print(f'❌ BodyReconstructor failed: {e}')

    try:
        from src.core.ai_enhancement import AIEnhancer
        ae = AIEnhancer()
        ai_enabled = hasattr(ae, 'enabled') and ae.enabled
        print(f'✅ AIEnhancer initialized (AI enhancement enabled: {ai_enabled})')
    except Exception as e:
        print(f'❌ AIEnhancer failed: {e}')

    try:
        from src.core.garment_fitting import GarmentFitter
        gf = GarmentFitter()
        has_pybullet = gf.physics_client is not None
        print(f'✅ GarmentFitter initialized (PyBullet available: {has_pybullet})')
    except Exception as e:
        print(f'❌ GarmentFitter failed: {e}')

    try:
        from src.core.rendering import PhotorealisticRenderer
        pr = PhotorealisticRenderer()
        has_blender = pr.blender_available
        print(f'✅ PhotorealisticRenderer initialized (Blender available: {has_blender})')
    except Exception as e:
        print(f'❌ PhotorealisticRenderer failed: {e}')

def analyze_package_sizes():
    """Analyze estimated package sizes"""
    print('\nEstimated package sizes:')
    estimated_sizes = {
        'torch': '~8GB',
        'torchvision': '~2GB', 
        'torchaudio': '~1GB',
        'diffusers': '~2GB',
        'transformers': '~2GB',
        'open3d': '~2GB',
        'pybullet': '~500MB',
        'mediapipe': '~300MB',
        'trimesh': '~100MB',
        'opencv-python': '~200MB'
    }
    
    for package, size in estimated_sizes.items():
        print(f'  {package}: {size}')
    
    print('\nTotal estimated size of heavy dependencies: ~18GB')
    print('Base Python 3.12-slim image: ~150MB')
    print('System dependencies: ~500MB')
    print('Other Python packages: ~1GB')
    print('Total current image size: ~20GB')

if __name__ == '__main__':
    test_minimal_requirements()
    test_3d_dependencies()
    analyze_package_sizes()
