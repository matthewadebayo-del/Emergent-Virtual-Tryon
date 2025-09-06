import os
import subprocess
import tempfile
from typing import Optional, Tuple, TYPE_CHECKING, Any

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None

if TYPE_CHECKING and BLENDER_AVAILABLE:
    import bpy
import logging
import numpy as np
from PIL import Image
import cv2

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    BLENDER_AVAILABLE = True
except ImportError:
    print("⚠️ Blender Python API not available")
    BLENDER_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("⚠️ Trimesh not available, using basic rendering fallback")
    TRIMESH_AVAILABLE = False

logger = logging.getLogger(__name__)

class FixedPhorealisticRenderer:
    """Fixed version of the photorealistic renderer addressing the DirectionalLight issue"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        logger.info("🎬 Initializing Fixed Renderer...")
        if BLENDER_AVAILABLE:
            self._setup_blender_fixed()
        
    def _setup_blender_fixed(self):
        """Fixed Blender setup addressing API issues"""
        try:
            logger.info("Setting up Blender with fixes...")
            
            if self.headless:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                logger.info("✅ Blender factory settings loaded")
            
            # Set render engine with error handling
            try:
                bpy.context.scene.render.engine = 'CYCLES'
                logger.info("✅ Cycles render engine set")
            except Exception as e:
                logger.warning(f"⚠️ Cycles not available, using EEVEE: {e}")
                bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            
            # Safe GPU setup with fallback
            self._setup_gpu_safe()
            
            # Set conservative render settings
            scene = bpy.context.scene
            scene.render.resolution_x = 512  # Start smaller for debugging
            scene.render.resolution_y = 512
            scene.render.resolution_percentage = 100
            
            if bpy.context.scene.render.engine == 'CYCLES':
                scene.cycles.samples = 32  # Low samples for debugging
                scene.cycles.device = 'CPU'  # Force CPU for reliability
            
            logger.info("✅ Fixed Blender setup complete")
            
        except Exception as e:
            logger.error(f"❌ Blender setup failed: {e}")
            raise
    
    def _setup_gpu_safe(self):
        """Safe GPU setup with proper fallbacks"""
        try:
            # Try GPU setup with error handling
            preferences = bpy.context.preferences
            cycles_preferences = preferences.addons['cycles'].preferences
            
            try:
                cycles_preferences.compute_device_type = 'CUDA'
                cycles_preferences.get_devices()
                
                cuda_devices = [d for d in cycles_preferences.devices if d.type == 'CUDA']
                if cuda_devices and len(cuda_devices) > 0:
                    for device in cuda_devices:
                        device.use = True
                    bpy.context.scene.cycles.device = 'GPU'
                    logger.info(f"✅ GPU rendering enabled: {len(cuda_devices)} devices")
                else:
                    bpy.context.scene.cycles.device = 'CPU'
                    logger.info("⚠️ No CUDA devices, using CPU")
                    
            except Exception as gpu_error:
                logger.warning(f"⚠️ GPU setup failed: {gpu_error}, using CPU")
                bpy.context.scene.cycles.device = 'CPU'
                
        except Exception as e:
            logger.warning(f"⚠️ Could not access cycles preferences: {e}")
    
    def clear_scene(self):
        """Safely clear all objects from scene"""
        try:
            # Select all objects
            bpy.ops.object.select_all(action='SELECT')
            # Delete selected objects
            bpy.ops.object.delete(use_global=False)
            
            # Also clear orphaned data
            for mesh in bpy.data.meshes:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            
            for material in bpy.data.materials:
                if material.users == 0:
                    bpy.data.materials.remove(material)
                    
            logger.info("✅ Scene cleared successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Scene clearing issue: {e}")
    
    def import_mesh_safe(self, mesh, name: str = "imported_mesh") -> Optional[Any]:
        """Safely import mesh with validation"""
        try:
            logger.info(f"Importing mesh: {name}")
            
            # Validate mesh data
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                logger.error(f"❌ Mesh {name} has no vertices")
                return None
                
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                logger.error(f"❌ Mesh {name} has no faces")
                return None
            
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()
            
            logger.info(f"Mesh {name}: {len(vertices)} vertices, {len(faces)} faces")
            
            # Create mesh data
            mesh_data = bpy.data.meshes.new(name)
            mesh_data.from_pydata(vertices, [], faces)
            mesh_data.update()
            
            # Validate created mesh
            if len(mesh_data.vertices) == 0:
                logger.error(f"❌ Blender mesh creation failed for {name}")
                return None
            
            # Create object
            obj = bpy.data.objects.new(name, mesh_data)
            bpy.context.collection.objects.link(obj)
            
            # Set object location to origin
            obj.location = (0, 0, 0)
            obj.rotation_euler = (0, 0, 0)
            obj.scale = (1, 1, 1)
            
            logger.info(f"✅ Mesh {name} imported at location {obj.location}")
            return obj
            
        except Exception as e:
            logger.error(f"❌ Failed to import mesh {name}: {e}")
            return None
    
    def create_simple_material(self, name: str, color: Tuple[float, float, float] = (0.8, 0.6, 0.4)) -> Any:
        """Create simple, reliable material"""
        try:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear existing nodes
            nodes.clear()
            
            # Create simple emission shader for visibility
            emission = nodes.new(type='ShaderNodeEmission')
            emission.inputs['Color'].default_value = (*color, 1.0)
            emission.inputs['Strength'].default_value = 1.0
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            links.new(emission.outputs['Emission'], output.inputs['Surface'])
            
            logger.info(f"✅ Created material: {name}")
            return mat
            
        except Exception as e:
            logger.error(f"❌ Material creation failed: {e}")
            # Return a basic material without nodes
            mat = bpy.data.materials.new(name=name)
            mat.diffuse_color = (*color, 1.0)
            return mat
    
    def setup_basic_lighting(self):
        """Setup simple, reliable lighting"""
        try:
            logger.info("Setting up basic lighting...")
            
            # Method 1: Try to add sun light
            try:
                bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
                sun = bpy.context.active_object
                sun.data.energy = 5  # Bright light
                logger.info("✅ Sun light added")
            except Exception as e:
                logger.warning(f"⚠️ Could not add sun light: {e}")
            
            # Method 2: Add world background lighting
            try:
                world = bpy.context.scene.world
                if not world:
                    world = bpy.data.worlds.new("World")
                    bpy.context.scene.world = world
                
                world.use_nodes = True
                if 'Background' in world.node_tree.nodes:
                    bg_node = world.node_tree.nodes['Background']
                    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White
                    bg_node.inputs['Strength'].default_value = 2.0  # Bright
                    logger.info("✅ World background lighting set")
                
            except Exception as e:
                logger.warning(f"⚠️ World lighting setup failed: {e}")
            
            # Method 3: Add area lights as backup
            try:
                # Front light
                bpy.ops.object.light_add(type='AREA', location=(2, -3, 2))
                front_light = bpy.context.active_object
                front_light.data.energy = 100
                front_light.data.size = 3
                
                # Fill light
                bpy.ops.object.light_add(type='AREA', location=(-2, -3, 1))
                fill_light = bpy.context.active_object
                fill_light.data.energy = 50
                fill_light.data.size = 2
                
                logger.info("✅ Area lights added")
                
            except Exception as e:
                logger.warning(f"⚠️ Area lights failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Lighting setup failed: {e}")
    
    def setup_camera_safe(self, distance: float = 4.0) -> Optional[Any]:
        """Setup camera with safe positioning"""
        try:
            logger.info("Setting up camera...")
            
            # Remove existing cameras
            for obj in bpy.context.scene.objects:
                if obj.type == 'CAMERA':
                    bpy.data.objects.remove(obj, do_unlink=True)
            
            # Add new camera
            bpy.ops.object.camera_add(location=(0, -distance, 1.5))
            camera = bpy.context.active_object
            
            # Point camera at origin with safe rotation
            camera.rotation_euler = (1.1, 0, 0)  # Look slightly down
            
            # Set as active camera
            bpy.context.scene.camera = camera
            
            # Set camera properties
            camera.data.lens = 50  # Standard lens
            camera.data.clip_start = 0.1
            camera.data.clip_end = 1000
            
            logger.info(f"✅ Camera set at location {camera.location}")
            return camera
            
        except Exception as e:
            logger.error(f"❌ Camera setup failed: {e}")
            return None
    
    def validate_scene_before_render(self) -> bool:
        """Validate scene is ready for rendering"""
        try:
            # Check for objects
            if len(bpy.context.scene.objects) == 0:
                logger.error("❌ No objects in scene")
                return False
            
            # Check for camera
            if not bpy.context.scene.camera:
                logger.error("❌ No camera in scene")
                return False
            
            # Check for lights
            lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
            if len(lights) == 0:
                logger.warning("⚠️ No lights in scene, but continuing...")
            
            # Log scene contents
            scene_objects = [f"{obj.name}({obj.type})" for obj in bpy.context.scene.objects]
            logger.info(f"Scene objects: {scene_objects}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Scene validation failed: {e}")
            return False
    
    def render_scene_debug(self, body_mesh, garment_mesh, output_path: str) -> bool:
        """Debug rendering with extensive error handling"""
        if not BLENDER_AVAILABLE:
            logger.warning("⚠️ Blender not available, using fallback")
            return False
            
        try:
            logger.info("🎬 Starting debug render...")
            
            # Clear scene
            self.clear_scene()
            
            # Import meshes
            logger.info("Importing body mesh...")
            body_obj = self.import_mesh_safe(body_mesh, "body")
            if not body_obj:
                logger.error("❌ Failed to import body mesh")
                return False
            
            logger.info("Importing garment mesh...")
            garment_obj = self.import_mesh_safe(garment_mesh, "garment")
            if not garment_obj:
                logger.error("❌ Failed to import garment mesh")
                return False
            
            # Create materials
            logger.info("Creating materials...")
            body_material = self.create_simple_material("body_mat", (0.9, 0.7, 0.6))  # Skin color
            garment_material = self.create_simple_material("garment_mat", (0.2, 0.4, 0.8))  # Blue garment
            
            # Apply materials
            if body_obj.data.materials:
                body_obj.data.materials[0] = body_material
            else:
                body_obj.data.materials.append(body_material)
                
            if garment_obj.data.materials:
                garment_obj.data.materials[0] = garment_material
            else:
                garment_obj.data.materials.append(garment_material)
            
            # Setup lighting
            logger.info("Setting up lighting...")
            self.setup_basic_lighting()
            
            # Setup camera
            logger.info("Setting up camera...")
            camera = self.setup_camera_safe()
            if not camera:
                logger.error("❌ Camera setup failed")
                return False
            
            # Validate scene
            if not self.validate_scene_before_render():
                logger.error("❌ Scene validation failed")
                return False
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Set render output
            bpy.context.scene.render.filepath = output_path
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            
            # Additional render settings
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.image_settings.color_depth = '8'
            
            logger.info(f"Rendering to: {output_path}")
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            # Verify output
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Render complete! Size: {file_size} bytes")
                
                # Basic validation - check if file is not too small
                if file_size < 1000:  # Less than 1KB is probably empty
                    logger.warning(f"⚠️ Rendered file seems too small: {file_size} bytes")
                    return False
                
                return True
            else:
                logger.error(f"❌ Render file not created: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Render failed with error: {e}")
            
            # Save debug blend file
            try:
                debug_blend_path = output_path.replace('.png', '_debug.blend')
                bpy.ops.wm.save_as_mainfile(filepath=debug_blend_path)
                logger.info(f"Debug blend file saved: {debug_blend_path}")
            except:
                pass
            
            return False

    def render_scene(
        self,
        body_mesh,
        garment_mesh,
        output_path: str,
        fabric_type: str = "cotton",
        fabric_color: Tuple[float, float, float] = (0.2, 0.3, 0.8),
    ) -> str:
        """Main render_scene method for compatibility with existing code"""
        success = self.render_scene_debug(body_mesh, garment_mesh, output_path)
        if success:
            return output_path
        else:
            fallback = SimpleFallbackRenderer()
            if fallback.create_simple_composite(body_mesh, garment_mesh, output_path):
                return output_path
            else:
                placeholder = Image.new("RGB", (512, 512), color=(220, 200, 180))
                placeholder.save(output_path)
                return output_path


class SimpleFallbackRenderer:
    """Simple fallback renderer using OpenCV when Blender fails"""
    
    def __init__(self):
        logger.info("🎨 Initializing Simple Fallback Renderer...")
    
    def create_simple_composite(self, body_mesh, garment_mesh, output_path: str, 
                              image_size: Tuple[int, int] = (512, 512)) -> bool:
        """Create simple composite image when 3D rendering fails"""
        try:
            logger.info("Creating simple composite...")
            
            # Create blank image
            img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw simple representation
            center_x, center_y = image_size[0] // 2, image_size[1] // 2
            
            # Draw body outline (simple)
            cv2.ellipse(img, (center_x, center_y + 50), (80, 150), 0, 0, 360, (200, 180, 160), -1)  # Body
            cv2.circle(img, (center_x, center_y - 120), 40, (220, 200, 180), -1)  # Head
            
            # Draw garment outline
            cv2.ellipse(img, (center_x, center_y + 20), (90, 100), 0, 0, 360, (50, 100, 200), -1)  # Garment
            
            # Add text
            cv2.putText(img, "Virtual Try-On Preview", (center_x - 120, image_size[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save image
            cv2.imwrite(output_path, img)
            
            if os.path.exists(output_path):
                logger.info(f"✅ Fallback composite created: {output_path}")
                return True
            else:
                logger.error("❌ Failed to save fallback composite")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fallback rendering failed: {e}")
            return False


class RenderingPipeline:
    """Complete fixed rendering pipeline - main class for compatibility"""
    
    def __init__(self):
        self.renderer = FixedPhorealisticRenderer()
        self.fallback_renderer = SimpleFallbackRenderer()
    
    def render_scene(
        self,
        body_mesh,
        garment_mesh,
        output_path: str,
        fabric_type: str = "cotton",
        fabric_color: Tuple[float, float, float] = (0.2, 0.3, 0.8),
    ) -> str:
        """Complete rendering pipeline with fallbacks"""
        
        # Stage 1: Try 3D rendering
        logger.info("🎬 Stage 1: 3D Rendering...")
        temp_render_path = output_path.replace('.png', '_temp.png')
        
        success = self.renderer.render_scene_debug(body_mesh, garment_mesh, temp_render_path)
        
        if not success:
            logger.warning("⚠️ 3D rendering failed, trying fallback...")
            
            # Stage 2: Fallback rendering
            logger.info("🎨 Stage 2: Fallback Rendering...")
            success = self.fallback_renderer.create_simple_composite(
                body_mesh, garment_mesh, temp_render_path
            )
            
            if not success:
                logger.error("❌ All rendering methods failed")
                placeholder = Image.new("RGB", (512, 512), color=(220, 200, 180))
                placeholder.save(output_path)
                return output_path
        
        try:
            if os.path.exists(temp_render_path):
                if temp_render_path != output_path:
                    import shutil
                    shutil.copy2(temp_render_path, output_path)
                    os.remove(temp_render_path)
        except:
            pass
        
        return output_path


PhotorealisticRenderer = FixedPhorealisticRenderer
