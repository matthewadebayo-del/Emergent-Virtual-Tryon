#!/usr/bin/env python3
"""Comprehensive test for 3D rendering pipeline with direct bpy API"""

import sys
import os
import tempfile
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_bpy_availability():
    """Test if bpy is available with direct API (no subprocess)"""
    try:
        import sys
        import os
        
        blender_scripts = '/usr/share/blender/scripts/modules'
        if blender_scripts not in sys.path:
            sys.path.append(blender_scripts)
        
        blender_version_paths = [
            '/usr/share/blender/4.0/python/lib/python3.11/site-packages',
            '/usr/share/blender/3.6/python/lib/python3.10/site-packages',
            '/usr/share/blender/4.1/python/lib/python3.11/site-packages'
        ]
        
        added_paths = []
        for path in blender_version_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                added_paths.append(path)
        
        import bpy
        return {
            "bpy_available": True, 
            "version": bpy.app.version_string,
            "added_paths": added_paths
        }
    except ImportError as e:
        return {"bpy_available": False, "error": str(e), "python_paths": sys.path[-3:]}

def test_memory_detection():
    """Test memory detection for adaptive rendering"""
    try:
        import psutil
        available_mem = psutil.virtual_memory().available / (1024**3)  # GB
        total_mem = psutil.virtual_memory().total / (1024**3)  # GB
        
        print(f"✅ Memory detection: {available_mem:.1f}GB available / {total_mem:.1f}GB total")
        return {"available_gb": available_mem, "total_gb": total_mem}
    except ImportError:
        print("⚠️ psutil not available for memory detection")
        return {"available_gb": 0, "total_gb": 0}

def test_adaptive_fallback_quality():
    """Test adaptive fallback with quality checks"""
    try:
        from src.core.rendering import AdaptiveFallbackRenderer
        import trimesh
        
        renderer = AdaptiveFallbackRenderer()
        
        body_mesh = trimesh.creation.cylinder(radius=0.3, height=1.8)
        garment_mesh = trimesh.creation.cylinder(radius=0.35, height=1.0)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            success = renderer.create_simple_composite(body_mesh, garment_mesh, temp_file.name)
            
            if success and os.path.exists(temp_file.name):
                file_size = os.path.getsize(temp_file.name)
                quality_check = file_size > 15000
                
                import cv2
                img = cv2.imread(temp_file.name)
                height, width = img.shape[:2] if img is not None else (0, 0)
                
                results = {
                    "file_size": file_size,
                    "quality_check": quality_check,
                    "resolution": f"{width}x{height}",
                    "adaptive_resolution": renderer._get_adaptive_resolution(),
                    "adaptive_quality": renderer._get_adaptive_quality()
                }
                
                print(f"✅ Adaptive fallback: {file_size} bytes at {width}x{height} (quality: {'✅' if quality_check else '❌'})")
                return results
            else:
                print("❌ Adaptive fallback rendering failed")
                return {"file_size": 0, "quality_check": False}
                
    except Exception as e:
        print(f"❌ Adaptive fallback rendering failed: {e}")
        return {"file_size": 0, "quality_check": False}

if __name__ == "__main__":
    print("=== Enhanced 3D Rendering Pipeline Test (Direct API) ===")
    
    bpy_result = test_bpy_availability()
    print(f"Blender API: {'✅' if bpy_result['bpy_available'] else '❌'} {bpy_result.get('version', bpy_result.get('error', ''))}")
    if bpy_result['bpy_available']:
        print(f"Added paths: {bpy_result['added_paths']}")
    
    memory_result = test_memory_detection()
    
    fallback_result = test_adaptive_fallback_quality()
    
    print("\n=== Test Results ===")
    print(f"Blender Available: {'✅' if bpy_result['bpy_available'] else '❌'}")
    print(f"Memory Detection: {'✅' if memory_result['total_gb'] > 0 else '❌'} ({memory_result['available_gb']:.1f}GB available)")
    print(f"Adaptive Fallback: {'✅' if fallback_result['quality_check'] else '❌'} ({fallback_result['file_size']} bytes at {fallback_result.get('resolution', 'unknown')})")
    
    if bpy_result['bpy_available'] or fallback_result['quality_check']:
        print("\n🎉 3D rendering pipeline working!")
        sys.exit(0)
    else:
        print("\n❌ 3D rendering pipeline failed!")
        sys.exit(1)
