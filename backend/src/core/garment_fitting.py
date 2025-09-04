from typing import Any, Dict

import numpy as np

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    print("⚠️ Trimesh not available, using basic garment fitting")
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import pybullet as p

    PYBULLET_AVAILABLE = True
except ImportError:
    print("⚠️ PyBullet not available, using basic garment fitting")
    PYBULLET_AVAILABLE = False
    p = None


class GarmentFitter:
    """3D Garment fitting with physics simulation"""

    def __init__(self):
        self.garment_templates = self._create_garment_templates()
        self.physics_client = None

        if PYBULLET_AVAILABLE and p is not None:
            try:
                self.physics_client = p.connect(p.DIRECT)
                p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
                print("✅ PyBullet physics simulation initialized")
            except Exception as e:
                print(f"⚠️ PyBullet initialization failed: {e}")
                self.physics_client = None

    def _create_garment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced garment templates following plan specifications"""
        templates = {
            "shirts": {
                "t_shirt": self._create_enhanced_tshirt_template(),
                "polo_shirt": self._create_polo_template(),
                "dress_shirt": self._create_dress_shirt_template(),
            },
            "pants": {
                "jeans": self._create_jeans_template(),
                "chinos": self._create_chinos_template(),
                "shorts": self._create_shorts_template(),
            },
            "dresses": {
                "casual_dress": self._create_casual_dress_template(),
            },
        }
        return templates

    def _create_enhanced_tshirt_template(self):
        """Create detailed t-shirt template with better geometry"""
        if not TRIMESH_AVAILABLE or trimesh is None:
            print("⚠️ Trimesh not available, returning basic template")
            return None

        body = trimesh.creation.cylinder(radius=0.25, height=0.6, sections=32)

        sleeve_left = trimesh.creation.cylinder(radius=0.08, height=0.25, sections=16)
        sleeve_left.apply_translation([-0.33, 0, 0.15])
        sleeve_left.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        sleeve_right = trimesh.creation.cylinder(radius=0.08, height=0.25, sections=16)
        sleeve_right.apply_translation([0.33, 0, 0.15])
        sleeve_right.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        neck = trimesh.creation.cylinder(radius=0.08, height=0.05, sections=16)
        neck.apply_translation([0, 0, 0.32])

        tshirt = trimesh.util.concatenate([body, sleeve_left, sleeve_right])
        try:
            tshirt = tshirt.difference(neck)
        except Exception:
            pass

        return tshirt

    def _create_polo_template(self) -> trimesh.Trimesh:
        """Create polo shirt template"""
        base = self._create_enhanced_tshirt_template()

        collar = trimesh.creation.cylinder(radius=0.09, height=0.03, sections=16)
        collar.apply_translation([0, 0, 0.33])

        try:
            polo = trimesh.util.concatenate([base, collar])
        except Exception:
            polo = base

        return polo

    def _create_dress_shirt_template(self) -> trimesh.Trimesh:
        """Create dress shirt template"""
        body = trimesh.creation.cylinder(radius=0.22, height=0.65, sections=32)

        sleeve_left = trimesh.creation.cylinder(radius=0.07, height=0.35, sections=16)
        sleeve_left.apply_translation([-0.30, 0, 0.15])
        sleeve_left.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        sleeve_right = trimesh.creation.cylinder(radius=0.07, height=0.35, sections=16)
        sleeve_right.apply_translation([0.30, 0, 0.15])
        sleeve_right.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        dress_shirt = trimesh.util.concatenate([body, sleeve_left, sleeve_right])
        return dress_shirt

    def _create_jeans_template(self) -> trimesh.Trimesh:
        """Create jeans template"""
        leg_left = trimesh.creation.cylinder(radius=0.12, height=0.8, sections=16)
        leg_left.apply_translation([-0.1, 0, -0.4])

        leg_right = trimesh.creation.cylinder(radius=0.12, height=0.8, sections=16)
        leg_right.apply_translation([0.1, 0, -0.4])

        waist = trimesh.creation.cylinder(radius=0.2, height=0.1, sections=16)
        waist.apply_translation([0, 0, 0.35])

        jeans = trimesh.util.concatenate([leg_left, leg_right, waist])
        return jeans

    def _create_chinos_template(self) -> trimesh.Trimesh:
        """Create chinos template"""
        return self._create_jeans_template()

    def _create_shorts_template(self) -> trimesh.Trimesh:
        """Create shorts template"""
        leg_left = trimesh.creation.cylinder(radius=0.12, height=0.4, sections=16)
        leg_left.apply_translation([-0.1, 0, -0.2])

        leg_right = trimesh.creation.cylinder(radius=0.12, height=0.4, sections=16)
        leg_right.apply_translation([0.1, 0, -0.2])

        waist = trimesh.creation.cylinder(radius=0.2, height=0.1, sections=16)
        waist.apply_translation([0, 0, 0.35])

        shorts = trimesh.util.concatenate([leg_left, leg_right, waist])
        return shorts

    def _create_casual_dress_template(self) -> trimesh.Trimesh:
        """Create casual dress template"""
        body = trimesh.creation.cylinder(radius=0.25, height=1.0, sections=32)

        sleeve_left = trimesh.creation.cylinder(radius=0.08, height=0.2, sections=16)
        sleeve_left.apply_translation([-0.33, 0, 0.3])
        sleeve_left.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        sleeve_right = trimesh.creation.cylinder(radius=0.08, height=0.2, sections=16)
        sleeve_right.apply_translation([0.33, 0, 0.3])
        sleeve_right.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )

        dress = trimesh.util.concatenate([body, sleeve_left, sleeve_right])
        return dress

    def fit_garment_to_body(
        self, body_mesh: trimesh.Trimesh, garment_type: str, garment_subtype: str
    ) -> trimesh.Trimesh:
        """Fit garment template to specific body with physics simulation"""
        if garment_type not in self.garment_templates:
            garment_type = "shirts"
        if garment_subtype not in self.garment_templates[garment_type]:
            garment_subtype = list(self.garment_templates[garment_type].keys())[0]

        template = self.garment_templates[garment_type][garment_subtype]
        body_measurements = self._analyze_body_measurements(body_mesh)
        scaled_garment = self._scale_garment_to_body(template, body_measurements)

        if self.physics_client is not None:
            try:
                fitted_garment = self._run_physics_simulation(body_mesh, scaled_garment)
            except Exception as e:
                print(f"⚠️ Physics simulation failed: {e}, using basic fitting")
                fitted_garment = scaled_garment
        else:
            fitted_garment = scaled_garment

        return fitted_garment

    def _analyze_body_measurements(
        self, body_mesh: trimesh.Trimesh
    ) -> Dict[str, float]:
        """Extract detailed body measurements from mesh"""
        vertices = body_mesh.vertices

        measurements = {
            "chest_width": float(np.max(vertices[:, 0]) - np.min(vertices[:, 0])),
            "waist_width": float(
                (np.max(vertices[:, 0]) - np.min(vertices[:, 0])) * 0.8
            ),
            "hip_width": float((np.max(vertices[:, 0]) - np.min(vertices[:, 0])) * 0.9),
            "torso_length": float(np.max(vertices[:, 2]) - np.min(vertices[:, 2])),
            "chest_circumference": float(
                self._calculate_circumference(vertices, "chest")
            ),
            "waist_circumference": float(
                self._calculate_circumference(vertices, "waist")
            ),
        }

        return measurements

    def _calculate_circumference(self, vertices: np.ndarray, region: str) -> float:
        """Calculate circumference at specific body region"""
        if region == "chest":
            z_level = np.percentile(vertices[:, 2], 75)
        elif region == "waist":
            z_level = np.percentile(vertices[:, 2], 50)
        else:
            z_level = np.percentile(vertices[:, 2], 25)

        level_vertices = vertices[np.abs(vertices[:, 2] - z_level) < 0.05]

        if len(level_vertices) > 0:
            radius = np.mean(
                np.sqrt(level_vertices[:, 0] ** 2 + level_vertices[:, 1] ** 2)
            )
            return 2 * np.pi * radius
        return 0.0

    def _scale_garment_to_body(
        self, garment: trimesh.Trimesh, measurements: Dict[str, float]
    ) -> trimesh.Trimesh:
        """Scale garment template to fit body measurements"""
        scaled_garment = garment.copy()

        scale_x = float(max(measurements["chest_width"] / 0.5, 0.8))
        scale_y = float(max(measurements["chest_width"] / 0.5, 0.8))
        scale_z = float(max(measurements["torso_length"] / 0.6, 0.8))

        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_x
        scale_matrix[1, 1] = scale_y
        scale_matrix[2, 2] = scale_z
        scaled_garment.apply_transform(scale_matrix)

        return scaled_garment

    def _run_physics_simulation(
        self, body_mesh: trimesh.Trimesh, garment_mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        """Run basic physics simulation for garment draping"""
        body_bounds = body_mesh.bounds
        garment_bounds = garment_mesh.bounds

        if np.any(garment_bounds[0] < body_bounds[1]) and np.any(
            garment_bounds[1] > body_bounds[0]
        ):
            expansion_factor = 1.05
            center = garment_mesh.centroid
            expanded_garment = garment_mesh.copy()
            expanded_garment.vertices = (
                expanded_garment.vertices - center
            ) * expansion_factor + center
            return expanded_garment

        return garment_mesh
