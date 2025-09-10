import tempfile
import os
import subprocess
import trimesh
import logging
from src.core.rendering import BlenderSubprocessRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_blender_rendering():
    """Debug script to isolate Blender rendering failure"""
    logger.info("Starting Blender rendering debug...")
    
    body_mesh = trimesh.creation.cylinder(radius=0.3, height=1.8)
    garment_mesh = trimesh.creation.cylinder(radius=0.35, height=1.0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        body_path = os.path.join(temp_dir, "body.obj")
        garment_path = os.path.join(temp_dir, "garment.obj")
        
        logger.info(f"Exporting meshes to {temp_dir}")
        body_mesh.export(body_path)
        garment_mesh.export(garment_path)
        
        logger.info(f"Body mesh exported: {os.path.exists(body_path)}, size: {os.path.getsize(body_path) if os.path.exists(body_path) else 0}")
        logger.info(f"Garment mesh exported: {os.path.exists(garment_path)}, size: {os.path.getsize(garment_path) if os.path.exists(garment_path) else 0}")
        
        output_path = os.path.join(temp_dir, "test_render.png")
        renderer = BlenderSubprocessRenderer()
        
        logger.info(f"Blender available: {renderer.blender_available}")
        
        if renderer.blender_available:
            logger.info("Testing Blender version...")
            try:
                result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)
                logger.info(f"Blender version output: {result.stdout}")
            except Exception as e:
                logger.error(f"Blender version check failed: {e}")
        
        success = renderer.render_with_subprocess(body_mesh, garment_mesh, output_path)
        
        logger.info(f"Rendering success: {success}")
        logger.info(f"Output file exists: {os.path.exists(output_path)}")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Output file size: {file_size} bytes")
            
            if file_size == 0:
                logger.error("Output file is 0 bytes - rendering failed")
            elif file_size < 1000:
                logger.warning(f"Output file is very small ({file_size} bytes) - may be placeholder")
            else:
                logger.info("Output file size looks good")
        else:
            logger.error("Output file was not created")

if __name__ == "__main__":
    debug_blender_rendering()
