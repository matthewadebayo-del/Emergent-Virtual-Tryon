import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    print("⚠️ Trimesh not available, using basic rendering fallback")
    TRIMESH_AVAILABLE = False


class PhotorealisticRenderer:
    """Photorealistic rendering using Blender Cycles with enhanced fallback"""

    def __init__(self):
        self.blender_executable = "blender"
        self.blender_available = self._check_blender_availability()

    def _check_blender_availability(self) -> bool:
        """Check if Blender is available in the system"""
        try:
            result = subprocess.run(
                [self.blender_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("✅ Blender available for photorealistic rendering")
                return True
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            pass

        print("⚠️ Blender not available, using enhanced fallback rendering")
        return False

    def render_scene(
        self,
        body_mesh,
        garment_mesh,
        output_path: str,
        fabric_type: str = "cotton",
        fabric_color: Tuple[float, float, float] = (0.2, 0.3, 0.8),
    ) -> str:
        """Complete rendering pipeline using Blender or enhanced fallback"""

        if self.blender_available:
            return self._render_with_blender(
                body_mesh, garment_mesh, output_path, fabric_type, fabric_color
            )
        else:
            return self._render_enhanced_fallback(
                body_mesh, garment_mesh, output_path, fabric_color
            )

    def _render_with_blender(
        self,
        body_mesh,
        garment_mesh,
        output_path: str,
        fabric_type: str,
        fabric_color: Tuple[float, float, float],
    ) -> str:
        """Render using Blender with Cycles engine"""
        temp_files = []

        try:
            if body_mesh is None or garment_mesh is None:
                print("⚠️ Mesh data not available, using fallback rendering")
                return self._render_enhanced_fallback(
                    body_mesh, garment_mesh, output_path, fabric_color
                )

            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as body_file:
                body_mesh.export(body_file.name)
                body_path = body_file.name
                temp_files.append(body_path)

            with tempfile.NamedTemporaryFile(
                suffix=".obj", delete=False
            ) as garment_file:
                garment_mesh.export(garment_file.name)
                garment_path = garment_file.name
                temp_files.append(garment_path)

            blender_script = self._create_enhanced_blender_script(
                body_path, garment_path, output_path, fabric_type, fabric_color
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_file.write(blender_script)
                script_path = script_file.name
                temp_files.append(script_path)

            cmd = [self.blender_executable, "--background", "--python", script_path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                print(f"⚠️ Blender rendering failed: {result.stderr}")
                return self._render_enhanced_fallback(
                    body_mesh, garment_mesh, output_path, fabric_color
                )

            print("✅ Blender photorealistic rendering complete")
            return output_path

        except Exception as e:
            print(f"⚠️ Blender rendering error: {e}")
            return self._render_enhanced_fallback(
                body_mesh, garment_mesh, output_path, fabric_color
            )
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def _render_enhanced_fallback(
        self,
        body_mesh,
        garment_mesh,
        output_path: str,
        fabric_color: Tuple[float, float, float],
    ) -> str:
        """Enhanced fallback rendering with better quality and debugging"""
        print("🎨 Using enhanced fallback rendering")

        try:
            if body_mesh is None:
                print("⚠️ Body mesh is None, creating placeholder")
                placeholder = Image.new("RGB", (1024, 1024), color=(200, 180, 160))
                placeholder.save(output_path)
                return output_path
                
            if garment_mesh is None:
                print("⚠️ Garment mesh is None, creating placeholder")
                placeholder = Image.new("RGB", (1024, 1024), color=fabric_color)
                placeholder.save(output_path)
                return output_path

            if TRIMESH_AVAILABLE:
                print(f"Body mesh: {len(body_mesh.vertices)} vertices, {len(body_mesh.faces)} faces")
                print(f"Garment mesh: {len(garment_mesh.vertices)} vertices, {len(garment_mesh.faces)} faces")
                
                if len(body_mesh.vertices) == 0 or len(garment_mesh.vertices) == 0:
                    print("⚠️ Empty mesh detected, creating placeholder")
                    placeholder = Image.new("RGB", (1024, 1024), color=fabric_color)
                    placeholder.save(output_path)
                    return output_path

                combined_mesh = trimesh.util.concatenate([body_mesh, garment_mesh])
                print(f"Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")

                scene = trimesh.Scene([combined_mesh])
                scene.camera.resolution = [1024, 1024]
                scene.camera.fov = [60, 60]

                scene.camera_transform = trimesh.transformations.compose_matrix(
                    translate=[0, -2.5, 1.2], angles=[np.radians(10), 0, 0]
                )

                scene.lights = [
                    trimesh.scene.lighting.DirectionalLight(
                        intensity=1.0
                    ),
                    trimesh.scene.lighting.DirectionalLight(
                        intensity=0.5
                    )
                ]

                png_data = scene.save_image(resolution=[1024, 1024], visible=True)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(png_data)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"✅ Enhanced fallback rendering complete: {output_path}")
                    return output_path
                else:
                    print("⚠️ Rendered file is empty or missing")
                    raise Exception("Rendered file is empty")

            else:
                print("⚠️ Trimesh not available, creating basic placeholder")
                placeholder = Image.new("RGB", (1024, 1024), color=fabric_color)
                placeholder.save(output_path)
                return output_path

        except Exception as e:
            print(f"⚠️ Enhanced fallback rendering failed: {e}")
            placeholder = Image.new("RGB", (1024, 1024), color=(220, 200, 180))  # Light skin tone
            placeholder.save(output_path)
            return output_path

    def _create_enhanced_blender_script(
        self,
        body_path: str,
        garment_path: str,
        output_path: str,
        fabric_type: str,
        fabric_color: Tuple[float, float, float],
    ) -> str:
        """Create enhanced Blender Python script with comprehensive debugging"""
        return f"""
import bpy
import bmesh
import os
from mathutils import Vector

print("=== Starting Blender Debug Render ===")

try:
    print("Clearing scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print("Setting render engine to Cycles...")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.resolution_percentage = 100

    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = True
    
    bpy.context.scene.cycles.device = 'CPU'
    print("Using CPU rendering for stability")

    print(f"Importing body mesh from: {body_path}")
    if os.path.exists("{body_path}"):
        bpy.ops.import_scene.obj(filepath="{body_path}")
        if bpy.context.selected_objects:
            body_obj = bpy.context.selected_objects[0]
            body_obj.name = "body"
            print(f"✅ Body mesh imported: {{len(body_obj.data.vertices)}} vertices")
        else:
            print("❌ No body object selected after import")
            raise Exception("Body import failed")
    else:
        print(f"❌ Body file not found: {body_path}")
        raise Exception("Body file missing")

    print(f"Importing garment mesh from: {garment_path}")
    if os.path.exists("{garment_path}"):
        bpy.ops.import_scene.obj(filepath="{garment_path}")
        if bpy.context.selected_objects:
            garment_obj = bpy.context.selected_objects[0]
            garment_obj.name = "garment"
            print(f"✅ Garment mesh imported: {{len(garment_obj.data.vertices)}} vertices")
        else:
            print("❌ No garment object selected after import")
            raise Exception("Garment import failed")
    else:
        print(f"❌ Garment file not found: {garment_path}")
        raise Exception("Garment file missing")

    print("Creating simple materials...")
    
    skin_mat = bpy.data.materials.new(name="DebugSkin")
    skin_mat.use_nodes = True
    skin_nodes = skin_mat.node_tree.nodes
    skin_nodes.clear()
    
    skin_emission = skin_nodes.new(type='ShaderNodeEmission')
    skin_emission.inputs['Color'].default_value = (0.8, 0.7, 0.6, 1.0)
    skin_emission.inputs['Strength'].default_value = 0.8
    
    skin_output = skin_nodes.new(type='ShaderNodeOutputMaterial')
    skin_mat.node_tree.links.new(skin_emission.outputs['Emission'], skin_output.inputs['Surface'])

    fabric_mat = bpy.data.materials.new(name="DebugFabric")
    fabric_mat.use_nodes = True
    fabric_nodes = fabric_mat.node_tree.nodes
    fabric_nodes.clear()
    
    fabric_emission = fabric_nodes.new(type='ShaderNodeEmission')
    fabric_emission.inputs['Color'].default_value = ({fabric_color[0]}, {fabric_color[1]}, {fabric_color[2]}, 1.0)
    fabric_emission.inputs['Strength'].default_value = 1.0
    
    fabric_output = fabric_nodes.new(type='ShaderNodeOutputMaterial')
    fabric_mat.node_tree.links.new(fabric_emission.outputs['Emission'], fabric_output.inputs['Surface'])

    body_obj.data.materials.append(skin_mat)
    garment_obj.data.materials.append(fabric_mat)
    print("✅ Materials applied")

    body_obj.location = (0, 0, 0)
    garment_obj.location = (0, 0, 0)

    print("Setting up lighting...")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3
    print("✅ Sun light added")

    print("Setting up camera...")
    bpy.ops.object.camera_add(location=(0, -3, 1))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0)
    bpy.context.scene.camera = camera
    print("✅ Camera positioned")

    print("Validating scene...")
    objects = bpy.context.scene.objects
    print(f"Scene objects: {{[obj.name for obj in objects]}}")
    
    if not bpy.context.scene.camera:
        raise Exception("No camera in scene")
    
    output_dir = os.path.dirname("{output_path}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {{output_dir}}")

    bpy.context.scene.render.filepath = "{output_path}"
    print(f"Render output: {output_path}")

    print("Starting render...")
    bpy.ops.render.render(write_still=True)
    
    if os.path.exists("{output_path}"):
        file_size = os.path.getsize("{output_path}")
        print(f"✅ Render complete! File size: {{file_size}} bytes")
    else:
        print("❌ Render failed - output file not found")
        raise Exception("Render output missing")

except Exception as e:
    print(f"❌ Blender script failed: {{e}}")
    
    debug_blend = "{output_path}".replace('.png', '_debug.blend')
    bpy.ops.wm.save_as_mainfile(filepath=debug_blend)
    print(f"Debug blend file saved: {{debug_blend}}")
    
    import bpy
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    
    mat = bpy.data.materials.new(name="FallbackMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = ({fabric_color[0]}, {fabric_color[1]}, {fabric_color[2]}, 1)
    cube.data.materials.append(mat)
    
    bpy.context.scene.render.filepath = "{output_path}"
    bpy.ops.render.render(write_still=True)
    print("Fallback render created")
    
    raise
"""
