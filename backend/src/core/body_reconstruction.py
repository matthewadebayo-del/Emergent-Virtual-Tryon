from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️ MediaPipe not available, using basic measurement extraction")
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    print("⚠️ Trimesh not available, using basic mesh generation")
    TRIMESH_AVAILABLE = False
    trimesh = None


class BodyReconstructor:
    """3D Body reconstruction using enhanced measurement extraction"""

    def __init__(self):
        try:
            from src.core.advanced_measurement_extractor import AdvancedMeasurementExtractor
            self.measurement_extractor = AdvancedMeasurementExtractor()
            print("✅ Enhanced measurement extractor initialized")
        except ImportError as e:
            print(f"⚠️ Enhanced measurement extractor not available: {e}")
            self.measurement_extractor = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.7,
            )
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7
            )
        else:
            self.mp_pose = None
            self.mp_hands = None

        self.smpl_model = self._load_smpl_model()

    def _load_smpl_model(self):
        """Load SMPL-X model with fallback"""
        try:
            print("⚠️ SMPL-X model not available, using basic mesh generation")
            return None
        except Exception as e:
            print(f"⚠️ Failed to load SMPL-X model: {e}")
            return None

    def extract_pose_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract 2D pose landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_image)

        if not results.pose_landmarks:
            print("⚠️ No pose detected in image, using fallback measurements")
            return {
                "landmarks": np.zeros(
                    (33, 4)
                ),  # 33 pose landmarks with x,y,z,visibility
                "segmentation_mask": None,
            }

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return {
            "landmarks": np.array(landmarks),
            "segmentation_mask": results.segmentation_mask,
        }

    def estimate_body_measurements(
        self, landmarks: np.ndarray, image_shape: Tuple[int, int], 
        image: Optional[np.ndarray] = None, reference_height_cm: Optional[float] = None
    ) -> Dict[str, float]:
        """Enhanced body measurements using AdvancedMeasurementExtractor"""
        
        if self.measurement_extractor is not None and image is not None:
            try:
                print("🎯 Using enhanced measurement extraction")
                
                reference_measurement = None
                if reference_height_cm:
                    reference_measurement = {'type': 'height', 'value_cm': reference_height_cm}
                
                # Extract enhanced measurements
                enhanced_measurements = self.measurement_extractor.extract_measurements_single_image(
                    image, reference_measurement
                )
                
                # Convert to legacy format for backward compatibility
                measurements = {
                    "height": enhanced_measurements.height_cm,
                    "weight": enhanced_measurements.weight_kg or self._estimate_weight_from_enhanced(enhanced_measurements),
                    "chest_width": enhanced_measurements.chest_circumference / 3.14159,  # Approximate width from circumference
                    "chest_cm": enhanced_measurements.chest_circumference,
                    "waist_width": enhanced_measurements.waist_circumference / 3.14159,
                    "waist_cm": enhanced_measurements.waist_circumference,
                    "hip_width": enhanced_measurements.hip_circumference / 3.14159,
                    "hips_cm": enhanced_measurements.hip_circumference,
                    "shoulder_width_cm": enhanced_measurements.shoulder_width,
                    "torso_length": enhanced_measurements.torso_length,
                    "confidence_score": np.mean(list(enhanced_measurements.confidence_scores.values())) if enhanced_measurements.confidence_scores else 0.85,
                    "measurement_source": "enhanced_ai_extraction",
                    "enhanced_measurements": self._measurements_to_dict(enhanced_measurements),  # Store full measurements
                }
                
                print(f"✅ Enhanced measurements extracted with {len(enhanced_measurements.confidence_scores)} confidence scores")
                return measurements
                
            except Exception as e:
                print(f"⚠️ Enhanced measurement extraction failed: {e}, falling back to legacy method")
        
        return self._legacy_estimate_body_measurements(landmarks, image_shape)
    
    def _estimate_weight_from_enhanced(self, measurements) -> float:
        """Estimate weight from enhanced measurements using BMI approximation"""
        if measurements.height_cm > 0:
            avg_circumference = (measurements.chest_circumference + measurements.waist_circumference) / 2
            estimated_bmi = 18.5 + (avg_circumference - 80) * 0.15
            weight_kg = estimated_bmi * (measurements.height_cm / 100) ** 2
            return max(45.0, min(120.0, weight_kg))  # Reasonable bounds
        return 70.0  # Default weight
    
    def _measurements_to_dict(self, measurements):
        """Convert BodyMeasurements to dict safely"""
        try:
            from dataclasses import asdict
            return asdict(measurements)
        except:
            return {
                "height_cm": measurements.height_cm,
                "chest_circumference": measurements.chest_circumference,
                "waist_circumference": measurements.waist_circumference,
                "hip_circumference": measurements.hip_circumference,
                "shoulder_width": measurements.shoulder_width,
                "torso_length": measurements.torso_length,
                "confidence_scores": measurements.confidence_scores or {}
            }
    
    def _legacy_estimate_body_measurements(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Legacy measurement method (original implementation)"""
        h, w = image_shape[:2]
        pixel_landmarks = landmarks.copy()

        if np.all(landmarks == 0):
            print("⚠️ Using fallback measurements - no pose landmarks available")
            return {
                "height_cm": 170.0,
                "weight_kg": 70.0,
                "chest_width": 50.0,
                "chest_cm": 95.0,
                "waist_width": 40.0,
                "waist_cm": 80.0,
                "hip_width": 45.0,
                "hips_cm": 100.0,
                "shoulder_width_cm": 45.0,
                "torso_length": 60.0,
                "confidence_score": 0.3,
                "measurement_source": "fallback_estimation",
            }

        pixel_landmarks[:, 0] *= w
        pixel_landmarks[:, 1] *= h

        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        NOSE = 0

        confidence_scores = {}

        if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[RIGHT_SHOULDER][3] > 0.7:
            shoulder_width_px = abs(
                pixel_landmarks[LEFT_SHOULDER][0] - pixel_landmarks[RIGHT_SHOULDER][0]
            )
            confidence_scores["shoulder_width"] = min(
                landmarks[LEFT_SHOULDER][3], landmarks[RIGHT_SHOULDER][3]
            )
        else:
            shoulder_width_px = 0
            confidence_scores["shoulder_width"] = 0.0

        if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[LEFT_HIP][3] > 0.7:
            torso_length_px = abs(
                pixel_landmarks[LEFT_SHOULDER][1] - pixel_landmarks[LEFT_HIP][1]
            )
            confidence_scores["torso_length"] = min(
                landmarks[LEFT_SHOULDER][3], landmarks[LEFT_HIP][3]
            )
        else:
            torso_length_px = 0
            confidence_scores["torso_length"] = 0.0

        if landmarks[LEFT_HIP][3] > 0.7 and landmarks[RIGHT_HIP][3] > 0.7:
            hip_width_px = abs(
                pixel_landmarks[LEFT_HIP][0] - pixel_landmarks[RIGHT_HIP][0]
            )
            confidence_scores["hip_width"] = min(
                landmarks[LEFT_HIP][3], landmarks[RIGHT_HIP][3]
            )
        else:
            hip_width_px = 0
            confidence_scores["hip_width"] = 0.0

        if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[LEFT_WRIST][3] > 0.7:
            left_arm_length_px = np.linalg.norm(
                pixel_landmarks[LEFT_SHOULDER][:2] - pixel_landmarks[LEFT_WRIST][:2]
            )
            left_arm_confidence = min(
                landmarks[LEFT_SHOULDER][3], landmarks[LEFT_WRIST][3]
            )
        else:
            left_arm_length_px = 0
            left_arm_confidence = 0.0

        if landmarks[RIGHT_SHOULDER][3] > 0.7 and landmarks[RIGHT_WRIST][3] > 0.7:
            right_arm_length_px = np.linalg.norm(
                pixel_landmarks[RIGHT_SHOULDER][:2] - pixel_landmarks[RIGHT_WRIST][:2]
            )
            right_arm_confidence = min(
                landmarks[RIGHT_SHOULDER][3], landmarks[RIGHT_WRIST][3]
            )
        else:
            right_arm_length_px = 0
            right_arm_confidence = 0.0

        if left_arm_confidence > right_arm_confidence:
            arm_length_px = left_arm_length_px
            confidence_scores["arm_length"] = left_arm_confidence
        else:
            arm_length_px = right_arm_length_px
            confidence_scores["arm_length"] = right_arm_confidence

        if landmarks[LEFT_HIP][3] > 0.7 and landmarks[LEFT_ANKLE][3] > 0.7:
            left_leg_length_px = np.linalg.norm(
                pixel_landmarks[LEFT_HIP][:2] - pixel_landmarks[LEFT_ANKLE][:2]
            )
            left_leg_confidence = min(landmarks[LEFT_HIP][3], landmarks[LEFT_ANKLE][3])
        else:
            left_leg_length_px = 0
            left_leg_confidence = 0.0

        if landmarks[RIGHT_HIP][3] > 0.7 and landmarks[RIGHT_ANKLE][3] > 0.7:
            right_leg_length_px = np.linalg.norm(
                pixel_landmarks[RIGHT_HIP][:2] - pixel_landmarks[RIGHT_ANKLE][:2]
            )
            right_leg_confidence = min(
                landmarks[RIGHT_HIP][3], landmarks[RIGHT_ANKLE][3]
            )
        else:
            right_leg_length_px = 0
            right_leg_confidence = 0.0

        if left_leg_confidence > right_leg_confidence:
            leg_length_px = left_leg_length_px
            confidence_scores["leg_length"] = left_leg_confidence
        else:
            leg_length_px = right_leg_length_px
            confidence_scores["leg_length"] = right_leg_confidence

        if (
            landmarks[NOSE][3] > 0.7
            and max(left_leg_confidence, right_leg_confidence) > 0.7
        ):
            if left_leg_confidence > right_leg_confidence:
                height_px = abs(
                    pixel_landmarks[NOSE][1] - pixel_landmarks[LEFT_ANKLE][1]
                )
            else:
                height_px = abs(
                    pixel_landmarks[NOSE][1] - pixel_landmarks[RIGHT_ANKLE][1]
                )
            confidence_scores["height"] = min(
                landmarks[NOSE][3], max(left_leg_confidence, right_leg_confidence)
            )
        else:
            height_px = 0
            confidence_scores["height"] = 0.0

        # Convert pixel measurements to real-world measurements
        if shoulder_width_px > 0 and confidence_scores["shoulder_width"] > 0.7:
            pixel_to_inch_ratio = 18.0 / shoulder_width_px
        elif height_px > 0 and confidence_scores["height"] > 0.7:
            pixel_to_inch_ratio = 68.0 / height_px
        else:
            pixel_to_inch_ratio = 68.0 / max(h, w)

        measurements = {
            "shoulder_width": (
                shoulder_width_px * pixel_to_inch_ratio
                if shoulder_width_px > 0
                else 18.0
            ),
            "chest_width": (
                shoulder_width_px * pixel_to_inch_ratio * 1.1
                if shoulder_width_px > 0
                else 20.0
            ),
            "waist_width": (
                hip_width_px * pixel_to_inch_ratio * 0.85 if hip_width_px > 0 else 15.0
            ),
            "hip_width": (
                hip_width_px * pixel_to_inch_ratio if hip_width_px > 0 else 18.0
            ),
            "torso_length": (
                torso_length_px * pixel_to_inch_ratio if torso_length_px > 0 else 24.0
            ),
            "arm_length": (
                arm_length_px * pixel_to_inch_ratio if arm_length_px > 0 else 24.0
            ),
            "leg_length": (
                leg_length_px * pixel_to_inch_ratio if leg_length_px > 0 else 32.0
            ),
            "height": height_px * pixel_to_inch_ratio if height_px > 0 else 68.0,
            "pixel_to_inch_ratio": pixel_to_inch_ratio,
            "confidence_scores": confidence_scores,
            "overall_confidence": (
                np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
            ),
        }

        measurements = self._validate_measurements(measurements)

        return measurements

    def _validate_measurements(
        self, measurements: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate measurements using anthropometric ratios"""
        height = measurements["height"]

        expected_shoulder = height * 0.245
        if (
            abs(measurements["shoulder_width"] - expected_shoulder)
            > expected_shoulder * 0.3
        ):
            measurements["shoulder_width"] = expected_shoulder
            measurements["confidence_scores"]["shoulder_width"] *= 0.7

        expected_chest = height * 0.30
        if abs(measurements["chest_width"] - expected_chest) > expected_chest * 0.3:
            measurements["chest_width"] = expected_chest
            measurements["confidence_scores"]["chest_width"] = (
                measurements["confidence_scores"].get("chest_width", 0.5) * 0.7
            )

        expected_waist = height * 0.24
        if abs(measurements["waist_width"] - expected_waist) > expected_waist * 0.4:
            measurements["waist_width"] = expected_waist
            measurements["confidence_scores"]["waist_width"] = (
                measurements["confidence_scores"].get("waist_width", 0.5) * 0.7
            )

        expected_hip = height * 0.28
        if abs(measurements["hip_width"] - expected_hip) > expected_hip * 0.3:
            measurements["hip_width"] = expected_hip
            measurements["confidence_scores"]["hip_width"] *= 0.7

        expected_arm = height * 0.365
        if abs(measurements["arm_length"] - expected_arm) > expected_arm * 0.3:
            measurements["arm_length"] = expected_arm
            measurements["confidence_scores"]["arm_length"] *= 0.7

        expected_leg = height * 0.475
        if abs(measurements["leg_length"] - expected_leg) > expected_leg * 0.3:
            measurements["leg_length"] = expected_leg
            measurements["confidence_scores"]["leg_length"] *= 0.7

        measurements["overall_confidence"] = np.mean(
            list(measurements["confidence_scores"].values())
        )

        return measurements

    def create_enhanced_body_mesh(
        self, measurements: Dict[str, float]
    ) -> trimesh.Trimesh:
        """Create enhanced body mesh with better proportions"""
        if self.smpl_model is not None:
            return self._generate_smpl_mesh(measurements)
        else:
            return self._create_enhanced_basic_mesh(measurements)

    def _create_enhanced_basic_mesh(
        self, measurements: Dict[str, float]
    ) -> trimesh.Trimesh:
        """Create enhanced basic body mesh with better proportions"""
        chest_radius = measurements["chest_width"] / 2 / 100
        waist_radius = measurements["waist_width"] / 2 / 100
        hip_radius = measurements["hip_width"] / 2 / 100
        torso_height = measurements["torso_length"] / 100

        torso = trimesh.creation.cylinder(
            radius=chest_radius, height=torso_height, sections=32
        )

        vertices = torso.vertices.copy()
        for i, vertex in enumerate(vertices):
            z_pos = vertex[2]
            if z_pos < 0:
                factor = abs(z_pos) / (torso_height / 2)
                target_radius = chest_radius + factor * (hip_radius - chest_radius)
                current_radius = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
                if current_radius > 0:
                    scale = target_radius / current_radius
                    vertices[i][0] *= scale
                    vertices[i][1] *= scale
            else:
                factor = z_pos / (torso_height / 2)
                target_radius = chest_radius + factor * (waist_radius - chest_radius)
                current_radius = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
                if current_radius > 0:
                    scale = target_radius / current_radius
                    vertices[i][0] *= scale
                    vertices[i][1] *= scale

        body_mesh = trimesh.Trimesh(vertices=vertices, faces=torso.faces)
        return body_mesh

    def process_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Complete pipeline: image bytes -> 3D body mesh"""
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        pose_data = self.extract_pose_landmarks(image)
        measurements = self.estimate_body_measurements(
            pose_data["landmarks"], image.shape[:2]
        )
        body_mesh = self.create_enhanced_body_mesh(measurements)

        return {
            "body_mesh": body_mesh,
            "measurements": measurements,
            "pose_landmarks": pose_data["landmarks"],
            "segmentation_mask": pose_data["segmentation_mask"],
            "original_image_shape": image.shape,
        }
