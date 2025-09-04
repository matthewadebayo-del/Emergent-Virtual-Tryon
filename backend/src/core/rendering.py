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
        """Enhanced fallback rendering with better quality"""
        print("🎨 Using enhanced fallback rendering")

        try:
            if TRIMESH_AVAILABLE:
                combined_mesh = trimesh.util.concatenate([body_mesh, garment_mesh])

                scene = trimesh.Scene([combined_mesh])
                scene.camera.resolution = [1024, 1024]
                scene.camera.fov = [60, 60]

                scene.camera_transform = trimesh.transformations.compose_matrix(
                    translate=[0, -3, 1.5], angles=[np.radians(15), 0, 0]
                )

                png_data = scene.save_image(resolution=[1024, 1024], visible=True)

                with open(output_path, "wb") as f:
                    f.write(png_data)

                print(f"✅ Enhanced fallback rendering complete: {output_path}")
                return output_path
            else:
                print("⚠️ Trimesh not available, creating basic placeholder")
                placeholder = Image.new("RGB", (1024, 1024), color=fabric_color)
                placeholder.save(output_path)
                return output_path

        except Exception as e:
            print(f"⚠️ Enhanced fallback rendering failed: {e}")
            placeholder = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
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
        """Create enhanced Blender Python script for photorealistic rendering"""
        return f"""
import bpy
import bmesh
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.resolution_percentage = 100

bpy.context.scene.cycles.samples = 128
bpy.context.scene.cycles.use_denoising = True

bpy.ops.import_scene.obj(filepath="{body_path}")
body_obj = bpy.context.selected_objects[0]
body_obj.name = "body"

bpy.ops.import_scene.obj(filepath="{garment_path}")
garment_obj = bpy.context.selected_objects[0]
garment_obj.name = "garment"

skin_mat = bpy.data.materials.new(name="Skin")
skin_mat.use_nodes = True
skin_nodes = skin_mat.node_tree.nodes
skin_nodes.clear()

skin_bsdf = skin_nodes.new(type='ShaderNodeBsdfPrincipled')
skin_bsdf.inputs['Base Color'].default_value = (0.8, 0.7, 0.6, 1.0)
skin_bsdf.inputs['Subsurface'].default_value = 0.15
skin_bsdf.inputs['Subsurface Color'].default_value = (0.9, 0.8, 0.7, 1.0)
skin_bsdf.inputs['Roughness'].default_value = 0.4

skin_output = skin_nodes.new(type='ShaderNodeOutputMaterial')
skin_mat.node_tree.links.new(skin_bsdf.outputs['BSDF'], skin_output.inputs['Surface'])

fabric_mat = bpy.data.materials.new(name="Fabric")
fabric_mat.use_nodes = True
fabric_nodes = fabric_mat.node_tree.nodes
fabric_nodes.clear()

fabric_bsdf = fabric_nodes.new(type='ShaderNodeBsdfPrincipled')
fabric_bsdf.inputs['Base Color'].default_value = (
    {fabric_color[0]}, {fabric_color[1]}, {fabric_color[2]}, 1.0
)

if "{fabric_type}" == "cotton":
    fabric_bsdf.inputs['Roughness'].default_value = 0.8
    fabric_bsdf.inputs['Sheen'].default_value = 0.1
elif "{fabric_type}" == "silk":
    fabric_bsdf.inputs['Roughness'].default_value = 0.3
    fabric_bsdf.inputs['Sheen'].default_value = 0.8
elif "{fabric_type}" == "denim":
    fabric_bsdf.inputs['Roughness'].default_value = 0.9
    fabric_bsdf.inputs['Metallic'].default_value = 0.05
else:
    fabric_bsdf.inputs['Roughness'].default_value = 0.7

fabric_output = fabric_nodes.new(type='ShaderNodeOutputMaterial')
fabric_mat.node_tree.links.new(
    fabric_bsdf.outputs['BSDF'], fabric_output.inputs['Surface']
)

body_obj.data.materials.append(skin_mat)
garment_obj.data.materials.append(fabric_mat)

bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
key_light = bpy.context.active_object
key_light.data.energy = 100
key_light.data.size = 2

bpy.ops.object.light_add(type='AREA', location=(-1, -2, 2))
fill_light = bpy.context.active_object
fill_light.data.energy = 50
fill_light.data.size = 1.5

bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
rim_light = bpy.context.active_object
rim_light.data.energy = 30
rim_light.data.size = 1

bpy.ops.object.camera_add(location=(0, -3, 1.5))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0)
bpy.context.scene.camera = camera

camera.data.lens = 50
camera.data.sensor_width = 36

bpy.context.scene.render.filepath = "{output_path}"
bpy.ops.render.render(write_still=True)
"""
