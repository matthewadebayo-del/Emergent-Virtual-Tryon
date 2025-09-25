"""
Enhanced 3D Garment Processing
Uses actual garment analysis data for realistic 3D model creation and material properties
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import io

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class Enhanced3DGarmentProcessor:
    """Enhanced 3D garment processing using actual visual analysis data"""
    
    def __init__(self):
        self.initialized = TRIMESH_AVAILABLE
        self.garment_templates = self._initialize_garment_templates()
        self.material_properties = self._initialize_material_properties()
        
    def _initialize_garment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize 3D mesh templates for different garment types"""
        return {
            "t-shirt": {
                "base_extents": [0.6, 0.8, 0.1],
                "sleeve_ratio": 0.6,
                "neckline_depth": 0.15,
                "fit_type": "regular"
            },
            "polo_shirt": {
                "base_extents": [0.6, 0.75, 0.12],
                "sleeve_ratio": 0.5,
                "neckline_depth": 0.12,
                "fit_type": "fitted",
                "collar": True
            },
            "dress_shirt": {
                "base_extents": [0.65, 0.85, 0.1],
                "sleeve_ratio": 1.0,
                "neckline_depth": 0.1,
                "fit_type": "tailored",
                "collar": True,
                "buttons": True
            },
            "jeans": {
                "base_extents": [0.5, 1.2, 0.08],
                "leg_taper": 0.8,
                "rise": "mid",
                "fit_type": "straight"
            },
            "chinos": {
                "base_extents": [0.48, 1.15, 0.07],
                "leg_taper": 0.75,
                "rise": "mid",
                "fit_type": "slim"
            },
            "blazer": {
                "base_extents": [0.7, 0.9, 0.15],
                "sleeve_ratio": 1.0,
                "lapel_width": 0.08,
                "fit_type": "structured",
                "buttons": True
            },
            "dress": {
                "base_extents": [0.6, 1.3, 0.12],
                "waist_definition": True,
                "skirt_flare": 1.2,
                "fit_type": "fitted"
            }
        }
    
    def _initialize_material_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize material property mappings for different fabric types"""
        return {
            "silk": {
                "roughness": 0.1,
                "specular": 0.9,
                "metallic": 0.0,
                "sheen": 0.8,
                "transparency": 0.05
            },
            "cotton": {
                "roughness": 0.4,
                "specular": 0.5,
                "metallic": 0.0,
                "sheen": 0.2,
                "transparency": 0.0
            },
            "wool": {
                "roughness": 0.8,
                "specular": 0.1,
                "metallic": 0.0,
                "sheen": 0.1,
                "transparency": 0.0
            },
            "denim": {
                "roughness": 0.6,
                "specular": 0.3,
                "metallic": 0.0,
                "sheen": 0.15,
                "transparency": 0.0
            },
            "synthetic": {
                "roughness": 0.3,
                "specular": 0.7,
                "metallic": 0.1,
                "sheen": 0.6,
                "transparency": 0.0
            },
            "leather": {
                "roughness": 0.2,
                "specular": 0.8,
                "metallic": 0.0,
                "sheen": 0.9,
                "transparency": 0.0
            }
        }
    
    def create_enhanced_garment_mesh(self, garment_analysis: Dict[str, Any], garment_type: str = "t-shirt") -> Dict[str, Any]:
        """Create 3D garment mesh using actual visual analysis data"""
        if not self.initialized:
            return self._create_fallback_mesh(garment_analysis, garment_type)
        
        print(f"[3D] Creating enhanced {garment_type} mesh with visual analysis")
        
        # Get garment template
        template = self.garment_templates.get(garment_type, self.garment_templates["t-shirt"])
        
        # Use actual silhouette data to adjust proportions
        silhouette = garment_analysis.get("silhouette", {})
        aspect_ratio = silhouette.get("aspect_ratio", 1.0)
        
        # Adjust mesh dimensions based on actual garment proportions
        base_extents = template["base_extents"].copy()
        if aspect_ratio > 1.2:  # Wide garment
            base_extents[0] *= min(aspect_ratio, 1.5)
        elif aspect_ratio < 0.8:  # Tall garment
            base_extents[1] *= min(1.0/aspect_ratio, 1.3)
        
        print(f"[3D] Adjusted dimensions based on aspect ratio {aspect_ratio}: {base_extents}")
        
        # Create base mesh
        if garment_type in ["jeans", "chinos"]:
            mesh = self._create_pants_mesh(base_extents, template, silhouette)
        elif garment_type == "dress":
            mesh = self._create_dress_mesh(base_extents, template, silhouette)
        else:
            mesh = self._create_top_mesh(base_extents, template, silhouette)
        
        # Apply material properties based on fabric analysis
        material_props = self._get_material_properties(garment_analysis)
        
        # Create texture mapping based on patterns
        texture_data = self._create_texture_mapping(garment_analysis)
        
        return {
            "mesh": mesh,
            "material_properties": material_props,
            "texture_data": texture_data,
            "garment_type": garment_type,
            "analysis_used": True,
            "dimensions": base_extents,
            "vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0
        }
    
    def _create_top_mesh(self, extents: list, template: Dict[str, Any], silhouette: Dict[str, Any]) -> trimesh.Trimesh:
        """Create mesh for tops (t-shirts, polos, shirts, blazers)"""
        width, height, depth = extents
        
        # Create main body
        body_mesh = trimesh.creation.box(extents=extents)
        
        # Add sleeves if specified
        if template.get("sleeve_ratio", 0) > 0:
            sleeve_length = width * template["sleeve_ratio"]
            sleeve_radius = depth * 0.8
            
            # Left sleeve
            left_sleeve = trimesh.creation.cylinder(
                radius=sleeve_radius,
                height=sleeve_length,
                sections=12
            )
            left_sleeve.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi/2, [0, 1, 0]
            ))
            left_sleeve.apply_translation([-width/2 - sleeve_length/2, height*0.3, 0])
            
            # Right sleeve
            right_sleeve = trimesh.creation.cylinder(
                radius=sleeve_radius,
                height=sleeve_length,
                sections=12
            )
            right_sleeve.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi/2, [0, 1, 0]
            ))
            right_sleeve.apply_translation([width/2 + sleeve_length/2, height*0.3, 0])
            
            # Combine meshes
            body_mesh = trimesh.util.concatenate([body_mesh, left_sleeve, right_sleeve])
        
        # Add collar if specified
        if template.get("collar", False):
            collar_mesh = self._create_collar(width, depth)
            collar_mesh.apply_translation([0, height/2, 0])
            body_mesh = trimesh.util.concatenate([body_mesh, collar_mesh])
        
        return body_mesh
    
    def _create_pants_mesh(self, extents: list, template: Dict[str, Any], silhouette: Dict[str, Any]) -> trimesh.Trimesh:
        """Create mesh for pants (jeans, chinos)"""
        width, height, depth = extents
        
        # Create waist/hip area
        waist_mesh = trimesh.creation.box(extents=[width, height*0.3, depth])
        waist_mesh.apply_translation([0, height*0.35, 0])
        
        # Create legs with taper
        leg_taper = template.get("leg_taper", 0.8)
        leg_width = width * 0.4
        leg_height = height * 0.7
        
        # Left leg
        left_leg = trimesh.creation.cylinder(
            radius=leg_width/2,
            height=leg_height,
            sections=16
        )
        # Apply taper by scaling bottom
        vertices = left_leg.vertices.copy()
        for i, vertex in enumerate(vertices):
            if vertex[2] < 0:  # Bottom half
                scale_factor = leg_taper + (1 - leg_taper) * (vertex[2] + leg_height/2) / (leg_height/2)
                vertices[i][0] *= scale_factor
                vertices[i][1] *= scale_factor
        left_leg.vertices = vertices
        left_leg.apply_translation([-width*0.2, -height*0.15, 0])
        
        # Right leg (mirror of left)
        right_leg = left_leg.copy()
        right_leg.apply_translation([width*0.4, 0, 0])
        
        # Combine meshes
        pants_mesh = trimesh.util.concatenate([waist_mesh, left_leg, right_leg])
        
        return pants_mesh
    
    def _create_dress_mesh(self, extents: list, template: Dict[str, Any], silhouette: Dict[str, Any]) -> trimesh.Trimesh:
        """Create mesh for dresses"""
        width, height, depth = extents
        
        # Create bodice (top part)
        bodice_height = height * 0.4
        bodice_mesh = trimesh.creation.box(extents=[width*0.8, bodice_height, depth])
        bodice_mesh.apply_translation([0, height*0.3, 0])
        
        # Create skirt with flare
        skirt_flare = template.get("skirt_flare", 1.2)
        skirt_height = height * 0.6
        
        # Create flared skirt using truncated cone
        skirt_mesh = trimesh.creation.cone(
            radius=width*skirt_flare/2,
            height=skirt_height,
            sections=24
        )
        # Truncate the top to create waist
        skirt_mesh = skirt_mesh.slice_plane([0, 0, 1], [0, 0, skirt_height*0.8])
        skirt_mesh.apply_translation([0, -height*0.1, 0])
        
        # Combine bodice and skirt
        dress_mesh = trimesh.util.concatenate([bodice_mesh, skirt_mesh])
        
        return dress_mesh
    
    def _create_collar(self, width: float, depth: float) -> trimesh.Trimesh:
        """Create collar mesh for shirts and blazers"""
        collar_width = width * 0.6
        collar_depth = depth * 1.5
        collar_height = depth * 0.3
        
        collar_mesh = trimesh.creation.box(extents=[collar_width, collar_height, collar_depth])
        return collar_mesh
    
    def _get_material_properties(self, garment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert fabric analysis to 3D material properties"""
        fabric_type = garment_analysis.get("fabric_type", "cotton")
        colors = garment_analysis.get("colors", {})
        texture_data = garment_analysis.get("texture", {})
        
        # Get base material properties
        base_props = self.material_properties.get(fabric_type, self.material_properties["cotton"])
        
        # Adjust properties based on texture analysis
        roughness_adjustment = texture_data.get("roughness", 0.5)
        adjusted_roughness = base_props["roughness"] * (0.5 + roughness_adjustment)
        
        # Get primary color
        primary_color = colors.get("primary", (128, 128, 128))
        if isinstance(primary_color, tuple) and len(primary_color) >= 3:
            # Normalize RGB to 0-1 range
            base_color = [c/255.0 for c in primary_color[:3]]
        else:
            base_color = [0.5, 0.5, 0.5]
        
        return {
            "base_color": base_color,
            "roughness": min(1.0, max(0.0, adjusted_roughness)),
            "specular": base_props["specular"],
            "metallic": base_props["metallic"],
            "sheen": base_props["sheen"],
            "transparency": base_props["transparency"],
            "fabric_type": fabric_type,
            "color_palette": colors.get("palette", [primary_color])
        }
    
    def _create_texture_mapping(self, garment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create texture mapping based on pattern analysis"""
        patterns = garment_analysis.get("patterns", {})
        pattern_type = patterns.get("type", "solid")
        colors = garment_analysis.get("colors", {})
        
        texture_data = {
            "pattern_type": pattern_type,
            "has_pattern": patterns.get("detected", False),
            "pattern_complexity": patterns.get("complexity_score", 0.0)
        }
        
        if pattern_type == "vertical_stripes":
            texture_data.update({
                "stripe_direction": "vertical",
                "stripe_frequency": patterns.get("vertical_peaks", 5),
                "stripe_colors": colors.get("palette", [(128, 128, 128)])[:2]
            })
        elif pattern_type == "horizontal_stripes":
            texture_data.update({
                "stripe_direction": "horizontal", 
                "stripe_frequency": patterns.get("horizontal_peaks", 5),
                "stripe_colors": colors.get("palette", [(128, 128, 128)])[:2]
            })
        elif pattern_type == "checkered":
            texture_data.update({
                "check_size": min(patterns.get("horizontal_peaks", 5), patterns.get("vertical_peaks", 5)),
                "check_colors": colors.get("palette", [(128, 128, 128)])[:2]
            })
        elif pattern_type == "print":
            texture_data.update({
                "print_complexity": patterns.get("complexity_score", 0.5),
                "print_colors": colors.get("palette", [(128, 128, 128)])
            })
        
        return texture_data
    
    def _create_fallback_mesh(self, garment_analysis: Dict[str, Any], garment_type: str) -> Dict[str, Any]:
        """Create fallback mesh when trimesh is not available"""
        print(f"[3D] Creating fallback mesh for {garment_type}")
        
        # Get basic dimensions
        template = self.garment_templates.get(garment_type, self.garment_templates["t-shirt"])
        base_extents = template["base_extents"]
        
        # Get material properties
        material_props = self._get_material_properties(garment_analysis)
        texture_data = self._create_texture_mapping(garment_analysis)
        
        return {
            "mesh": None,  # No actual mesh
            "material_properties": material_props,
            "texture_data": texture_data,
            "garment_type": garment_type,
            "analysis_used": True,
            "dimensions": base_extents,
            "vertices": 0,
            "faces": 0,
            "fallback": True
        }
    
    def apply_physics_properties(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physics properties based on fabric type for realistic simulation"""
        fabric_type = mesh_data["material_properties"]["fabric_type"]
        
        # Define physics properties for different fabrics
        physics_properties = {
            "silk": {
                "stiffness": 0.2,
                "damping": 0.8,
                "mass_density": 0.1,
                "air_resistance": 0.9
            },
            "cotton": {
                "stiffness": 0.5,
                "damping": 0.6,
                "mass_density": 0.3,
                "air_resistance": 0.5
            },
            "wool": {
                "stiffness": 0.7,
                "damping": 0.4,
                "mass_density": 0.4,
                "air_resistance": 0.3
            },
            "denim": {
                "stiffness": 0.9,
                "damping": 0.3,
                "mass_density": 0.6,
                "air_resistance": 0.2
            },
            "synthetic": {
                "stiffness": 0.4,
                "damping": 0.7,
                "mass_density": 0.2,
                "air_resistance": 0.6
            }
        }
        
        fabric_physics = physics_properties.get(fabric_type, physics_properties["cotton"])
        mesh_data["physics_properties"] = fabric_physics
        
        print(f"[3D] Applied {fabric_type} physics properties: stiffness={fabric_physics['stiffness']}")
        
        return mesh_data
    
    def generate_procedural_texture(self, texture_data: Dict[str, Any], resolution: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
        """Generate procedural texture based on pattern analysis"""
        pattern_type = texture_data.get("pattern_type", "solid")
        width, height = resolution
        
        if pattern_type == "solid":
            return None  # Use base color only
        
        # Create base texture array
        texture = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        if pattern_type == "vertical_stripes":
            stripe_freq = texture_data.get("stripe_frequency", 5)
            stripe_colors = texture_data.get("stripe_colors", [(255, 255, 255), (0, 0, 0)])
            
            stripe_width = width // stripe_freq
            for i in range(0, width, stripe_width * 2):
                color = stripe_colors[0] if len(stripe_colors) > 0 else (255, 255, 255)
                texture[:, i:i+stripe_width] = color[:3]
                
        elif pattern_type == "horizontal_stripes":
            stripe_freq = texture_data.get("stripe_frequency", 5)
            stripe_colors = texture_data.get("stripe_colors", [(255, 255, 255), (0, 0, 0)])
            
            stripe_height = height // stripe_freq
            for i in range(0, height, stripe_height * 2):
                color = stripe_colors[0] if len(stripe_colors) > 0 else (255, 255, 255)
                texture[i:i+stripe_height, :] = color[:3]
                
        elif pattern_type == "checkered":
            check_size = max(1, width // texture_data.get("check_size", 8))
            check_colors = texture_data.get("check_colors", [(255, 255, 255), (0, 0, 0)])
            
            for y in range(0, height, check_size):
                for x in range(0, width, check_size):
                    color_idx = ((x // check_size) + (y // check_size)) % 2
                    color = check_colors[color_idx] if color_idx < len(check_colors) else (128, 128, 128)
                    texture[y:y+check_size, x:x+check_size] = color[:3]
        
        return texture
    
    def process_garment_with_analysis(
        self, 
        garment_analysis: Dict[str, Any],
        customer_analysis: Dict[str, Any],
        fitting_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process garment using both customer and garment analysis data"""
        
        try:
            # Determine garment type from analysis or fitting data
            garment_type = fitting_data.get("garment_type", "t-shirt")
            
            # Create garment mesh with analysis data
            garment_mesh = self.create_enhanced_garment_mesh(garment_analysis, garment_type)
            
            # Apply customer-specific fitting
            fitted_mesh = self._apply_customer_fitting(garment_mesh, customer_analysis, fitting_data)
            
            # Render final image
            rendered_image = self._render_fitted_garment(fitted_mesh, customer_analysis)
            
            return {
                "success": True,
                "rendered_image": rendered_image,
                "mesh_data": fitted_mesh,
                "processing_info": {
                    "garment_type": garment_type,
                    "colors_applied": garment_analysis.get("dominant_colors", []),
                    "fabric_type": garment_analysis.get("fabric_type", "cotton"),
                    "customer_measurements": customer_analysis.get("measurements", {})
                }
            }
            
        except Exception as e:
            print(f"[3D] Garment processing with analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _apply_customer_fitting(
        self, 
        garment_mesh: Dict[str, Any], 
        customer_analysis: Dict[str, Any], 
        fitting_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply customer-specific fitting to garment mesh"""
        
        # Get scaling factors
        scale_factors = fitting_data.get("scale_factors", {})
        positioning = fitting_data.get("positioning", {})
        
        # Apply scaling to mesh if available
        if garment_mesh.get("mesh") and hasattr(garment_mesh["mesh"], "vertices"):
            primary_scale = scale_factors.get("primary", 1.0)
            garment_mesh["mesh"].vertices *= primary_scale
        
        # Create fitted mesh data
        fitted_mesh = garment_mesh.copy()
        fitted_mesh["positioning"] = positioning
        fitted_mesh["scaling"] = scale_factors
        
        return fitted_mesh
    
    def _render_fitted_garment(
        self, 
        fitted_mesh: Dict[str, Any], 
        customer_analysis: Dict[str, Any]
    ) -> Image.Image:
        """Render the fitted garment on customer"""
        from PIL import ImageDraw
        
        # Get garment colors and properties
        material_props = fitted_mesh.get("material_properties", {})
        base_color = material_props.get("base_color", [0.5, 0.5, 0.5])
        garment_type = fitted_mesh.get("garment_type", "t-shirt")
        
        # Convert to RGB 0-255
        rgb_color = tuple(int(c * 255) for c in base_color[:3])
        
        # Create base image with person silhouette
        width, height = 512, 512
        rendered = Image.new("RGB", (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(rendered)
        
        # Draw person silhouette
        person_center = (width // 2, height // 2 + 50)
        person_width = 120
        person_height = 200
        
        # Body outline
        draw.ellipse([
            person_center[0] - person_width//2, person_center[1] - person_height//2,
            person_center[0] + person_width//2, person_center[1] + person_height//2
        ], fill=(220, 200, 180), outline=(200, 180, 160))
        
        # Draw garment based on type
        if garment_type in ["t-shirt", "polo_shirt", "dress_shirt"]:
            self._draw_shirt(draw, person_center, rgb_color, fitted_mesh)
        elif garment_type in ["jeans", "chinos"]:
            self._draw_pants(draw, person_center, rgb_color, fitted_mesh)
        elif garment_type == "dress":
            self._draw_dress(draw, person_center, rgb_color, fitted_mesh)
        elif garment_type == "blazer":
            self._draw_blazer(draw, person_center, rgb_color, fitted_mesh)
        
        return rendered
    
    def _draw_shirt(self, draw, center, color, mesh_data):
        """Draw shirt on person"""
        x, y = center
        # Main shirt body
        draw.rectangle([x-60, y-80, x+60, y+40], fill=color, outline=tuple(max(0, c-20) for c in color))
        # Sleeves
        draw.rectangle([x-85, y-80, x-60, y-30], fill=color)
        draw.rectangle([x+60, y-80, x+85, y-30], fill=color)
    
    def _draw_pants(self, draw, center, color, mesh_data):
        """Draw pants on person"""
        x, y = center
        # Waist area
        draw.rectangle([x-40, y-20, x+40, y+20], fill=color, outline=tuple(max(0, c-20) for c in color))
        # Left leg
        draw.rectangle([x-35, y+20, x-5, y+120], fill=color)
        # Right leg  
        draw.rectangle([x+5, y+20, x+35, y+120], fill=color)
    
    def _draw_dress(self, draw, center, color, mesh_data):
        """Draw dress on person"""
        x, y = center
        # Bodice
        draw.rectangle([x-50, y-80, x+50, y], fill=color, outline=tuple(max(0, c-20) for c in color))
        # Skirt (flared)
        draw.polygon([(x-50, y), (x+50, y), (x+70, y+100), (x-70, y+100)], fill=color)
    
    def _draw_blazer(self, draw, center, color, mesh_data):
        """Draw blazer on person"""
        x, y = center
        # Main blazer body
        draw.rectangle([x-65, y-90, x+65, y+30], fill=color, outline=tuple(max(0, c-30) for c in color))
        # Sleeves
        draw.rectangle([x-90, y-90, x-65, y-40], fill=color)
        draw.rectangle([x+65, y-90, x+90, y-40], fill=color)
        # Lapels
        draw.polygon([(x-10, y-90), (x-30, y-70), (x-10, y-50)], fill=tuple(max(0, c-10) for c in color))