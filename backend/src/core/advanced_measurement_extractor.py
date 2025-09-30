import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from PIL import Image
import math
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("[WARN] MediaPipe not available in AdvancedMeasurementExtractor - using fallback mode")

logger = logging.getLogger(__name__)

@dataclass
class BodyMeasurements:
    """Complete body measurements data structure"""
    
    # Basic info
    height_cm: float = 0.0
    weight_kg: Optional[float] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    
    # Head measurements
    head_circumference: float = 0.0
    neck_circumference: float = 0.0
    
    # Upper body measurements
    shoulder_width: float = 0.0
    chest_circumference: float = 0.0
    bust_circumference: float = 0.0  # For women
    underbust_circumference: float = 0.0  # For women
    waist_circumference: float = 0.0
    arm_length: float = 0.0
    forearm_length: float = 0.0
    bicep_circumference: float = 0.0
    wrist_circumference: float = 0.0
    
    # Lower body measurements
    hip_circumference: float = 0.0
    thigh_circumference: float = 0.0
    knee_circumference: float = 0.0
    calf_circumference: float = 0.0
    ankle_circumference: float = 0.0
    inseam_length: float = 0.0
    outseam_length: float = 0.0
    rise_length: float = 0.0  # Waist to crotch
    
    # Torso measurements
    torso_length: float = 0.0
    back_length: float = 0.0
    sleeve_length: float = 0.0
    
    # Confidence scores for each measurement
    confidence_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}

# Garment-specific measurement mappings
GARMENT_MEASUREMENTS = {
    'shirts': {
        'primary': ['chest_circumference', 'shoulder_width', 'arm_length', 'torso_length'],
        'secondary': ['neck_circumference', 'waist_circumference', 'sleeve_length'],
        'optional': ['bicep_circumference', 'wrist_circumference']
    },
    'blouses': {
        'primary': ['bust_circumference', 'shoulder_width', 'arm_length', 'torso_length'],
        'secondary': ['underbust_circumference', 'waist_circumference', 'sleeve_length'],
        'optional': ['bicep_circumference', 'wrist_circumference']
    },
    'pants': {
        'primary': ['waist_circumference', 'hip_circumference', 'inseam_length', 'thigh_circumference'],
        'secondary': ['rise_length', 'outseam_length', 'knee_circumference'],
        'optional': ['calf_circumference', 'ankle_circumference']
    },
    'jeans': {
        'primary': ['waist_circumference', 'hip_circumference', 'inseam_length', 'thigh_circumference'],
        'secondary': ['rise_length', 'outseam_length'],
        'optional': ['knee_circumference', 'calf_circumference']
    },
    'dresses': {
        'primary': ['bust_circumference', 'waist_circumference', 'hip_circumference', 'torso_length'],
        'secondary': ['shoulder_width', 'arm_length', 'back_length'],
        'optional': ['underbust_circumference', 'thigh_circumference']
    },
    'jackets': {
        'primary': ['chest_circumference', 'shoulder_width', 'arm_length', 'torso_length'],
        'secondary': ['waist_circumference', 'sleeve_length', 'back_length'],
        'optional': ['bicep_circumference', 'neck_circumference']
    },
    'skirts': {
        'primary': ['waist_circumference', 'hip_circumference'],
        'secondary': ['outseam_length'],
        'optional': ['thigh_circumference']
    }
}

class AdvancedMeasurementExtractor:
    """Advanced body measurement extraction from images"""
    
    def __init__(self):
        self.initialized = MEDIAPIPE_AVAILABLE
        
        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.7
            )
            
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
        else:
            self.mp_pose = None
            self.mp_hands = None
        
        # Anthropometric ratios (average human proportions)
        self.anthropometric_ratios = {
            'head_height_to_total': 0.125,     # Head is ~1/8 of total height
            'shoulder_width_to_height': 0.25,  # Shoulder width ~1/4 height
            'arm_span_to_height': 1.0,         # Arm span ≈ height
            'torso_to_height': 0.52,           # Torso ~52% of height
            'leg_to_height': 0.48,             # Legs ~48% of height
            'waist_to_chest_ratio': 0.8,       # Waist ~80% of chest (men)
            'hip_to_waist_ratio': 1.1,         # Hip ~110% of waist (women)
        }
        
    def extract_measurements_single_image(self, image: np.ndarray, 
                                        reference_measurement: Optional[Dict] = None) -> BodyMeasurements:
        """
        Extract measurements from single image with reference scaling
        
        Args:
            image: Input image as numpy array
            reference_measurement: Dict with known measurement for scaling
                                  e.g., {'type': 'height', 'value_cm': 170}
        """
        
        if not self.initialized:
            # Return fallback measurements when MediaPipe is not available
            return self._create_fallback_measurements(reference_measurement)
        
        # Extract pose landmarks
        pose_data = self._extract_pose_landmarks(image)
        if not pose_data:
            return self._create_fallback_measurements(reference_measurement)
        
        landmarks = pose_data['landmarks']
        segmentation = pose_data['segmentation']
        
        # Calculate scale factor
        scale_factor = self._calculate_scale_factor(landmarks, image.shape, reference_measurement)
        
        # Extract measurements
        measurements = BodyMeasurements()
        
        # Basic measurements
        measurements.height_cm = self._calculate_height(landmarks, scale_factor)
        measurements.shoulder_width = self._calculate_shoulder_width(landmarks, scale_factor)
        measurements.arm_length = self._calculate_arm_length(landmarks, scale_factor)
        
        # Circumference measurements (estimated from width/depth ratios)
        measurements.chest_circumference = self._estimate_chest_circumference(landmarks, scale_factor)
        measurements.waist_circumference = self._estimate_waist_circumference(landmarks, scale_factor)
        measurements.hip_circumference = self._estimate_hip_circumference(landmarks, scale_factor)
        
        # Length measurements
        measurements.torso_length = self._calculate_torso_length(landmarks, scale_factor)
        measurements.inseam_length = self._calculate_inseam_length(landmarks, scale_factor)
        
        # Limb measurements
        measurements.thigh_circumference = self._estimate_thigh_circumference(landmarks, scale_factor)
        measurements.bicep_circumference = self._estimate_bicep_circumference(landmarks, scale_factor)
        
        # Calculate confidence scores
        measurements.confidence_scores = self._calculate_confidence_scores(landmarks, segmentation)
        
        return measurements
    
    def extract_measurements_multi_view(self, images: List[np.ndarray], 
                                      reference_measurement: Optional[Dict] = None) -> BodyMeasurements:
        """
        Extract more accurate measurements from multiple views
        
        Args:
            images: List of images from different angles [front, side, back]
            reference_measurement: Known measurement for scaling
        """
        
        measurements_list = []
        confidence_weights = []
        
        for i, image in enumerate(images):
            try:
                measurements = self.extract_measurements_single_image(image, reference_measurement)
                measurements_list.append(measurements)
                
                # Weight based on view quality
                avg_confidence = np.mean(list(measurements.confidence_scores.values()))
                confidence_weights.append(avg_confidence)
                
            except Exception as e:
                logger.warning(f"Failed to extract measurements from image {i}: {e}")
                continue
        
        if not measurements_list:
            raise ValueError("Could not extract measurements from any image")
        
        # Combine measurements with weighted average
        final_measurements = self._combine_measurements(measurements_list, confidence_weights)
        
        return final_measurements
    
    def _extract_pose_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """Extract pose landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to normalized coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return {
            'landmarks': np.array(landmarks),
            'segmentation': results.segmentation_mask
        }
    
    def _calculate_scale_factor(self, landmarks: np.ndarray, image_shape: Tuple, 
                              reference: Optional[Dict] = None) -> float:
        """Calculate pixel-to-cm scale factor"""
        
        if reference and reference.get('type') == 'height':
            # Use provided height as reference
            pixel_height = self._calculate_height_pixels(landmarks, image_shape)
            real_height_cm = reference['value_cm']
            return real_height_cm / pixel_height
        
        elif reference and reference.get('type') == 'object':
            # Use known object in image as reference
            # Implementation would depend on object detection
            pass
        
        else:
            # Use average human proportions (less accurate)
            # Assume average height of 170cm
            pixel_height = self._calculate_height_pixels(landmarks, image_shape)
            estimated_height_cm = 170.0  # Default assumption
            return estimated_height_cm / pixel_height
    
    def _calculate_height_pixels(self, landmarks: np.ndarray, image_shape: Tuple) -> float:
        """Calculate height in pixels"""
        # Top of head to bottom of feet
        nose = landmarks[0]  # Nose as head reference
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Use lower ankle
        bottom_point = left_ankle if left_ankle[1] > right_ankle[1] else right_ankle
        
        # Convert normalized coordinates to pixels
        head_y = nose[1] * image_shape[0]
        foot_y = bottom_point[1] * image_shape[0]
        
        height_pixels = abs(foot_y - head_y)
        return height_pixels
    
    def _calculate_height(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Calculate height in cm"""
        # Implementation details for height calculation
        height_pixels = self._calculate_height_pixels(landmarks, (1, 1))  # Normalized
        return height_pixels * scale_factor
    
    def _calculate_shoulder_width(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Calculate shoulder width"""
        left_shoulder = landmarks[11]   # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        
        # Calculate distance in normalized coordinates
        width_normalized = abs(right_shoulder[0] - left_shoulder[0])
        
        # Convert to cm
        return width_normalized * scale_factor
    
    def _calculate_arm_length(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Calculate arm length (shoulder to wrist)"""
        # Use right arm as reference
        shoulder = landmarks[12]  # Right shoulder
        elbow = landmarks[14]     # Right elbow
        wrist = landmarks[16]     # Right wrist
        
        # Calculate total arm length
        upper_arm = euclidean([shoulder[0], shoulder[1]], [elbow[0], elbow[1]])
        forearm = euclidean([elbow[0], elbow[1]], [wrist[0], wrist[1]])
        
        total_length_normalized = upper_arm + forearm
        return total_length_normalized * scale_factor
    
    def _estimate_chest_circumference(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Estimate chest circumference from shoulder width and depth"""
        shoulder_width = self._calculate_shoulder_width(landmarks, scale_factor)
        
        # Use anthropometric estimation
        # Chest circumference ≈ 2.7 × shoulder width (empirical formula)
        chest_circumference = shoulder_width * 2.7
        
        return chest_circumference
    
    def _estimate_waist_circumference(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Estimate waist circumference"""
        # Find waist width (approximate from hip landmarks)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        waist_width_normalized = abs(right_hip[0] - left_hip[0]) * 0.8  # Waist is narrower than hips
        waist_width_cm = waist_width_normalized * scale_factor
        
        # Estimate circumference from width (assuming elliptical cross-section)
        # Circumference ≈ π × (width + depth) / 2
        # Approximate depth as 0.7 × width for average body
        estimated_depth = waist_width_cm * 0.7
        circumference = math.pi * (waist_width_cm + estimated_depth) / 2
        
        return circumference
    
    def _estimate_hip_circumference(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Estimate hip circumference"""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        hip_width_normalized = abs(right_hip[0] - left_hip[0])
        hip_width_cm = hip_width_normalized * scale_factor
        
        # Estimate circumference (hips are typically wider and rounder than waist)
        estimated_depth = hip_width_cm * 0.8
        circumference = math.pi * (hip_width_cm + estimated_depth) / 2
        
        return circumference
    
    def _calculate_torso_length(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Calculate torso length (shoulder to waist)"""
        shoulder = landmarks[12]  # Right shoulder
        hip = landmarks[24]       # Right hip
        
        torso_length_normalized = abs(hip[1] - shoulder[1])
        return torso_length_normalized * scale_factor
    
    def _calculate_inseam_length(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Calculate inseam length (crotch to ankle)"""
        hip = landmarks[24]    # Right hip (approximate crotch)
        ankle = landmarks[28]  # Right ankle
        
        inseam_normalized = abs(ankle[1] - hip[1])
        return inseam_normalized * scale_factor
    
    def _estimate_thigh_circumference(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Estimate thigh circumference"""
        hip = landmarks[24]   # Right hip
        knee = landmarks[26]  # Right knee
        
        # Estimate thigh width at widest point (upper thigh)
        thigh_width_normalized = abs(hip[0] - knee[0]) * 0.6  # Approximate width
        thigh_width_cm = thigh_width_normalized * scale_factor
        
        # Estimate circumference
        estimated_depth = thigh_width_cm * 0.9  # Thighs are quite round
        circumference = math.pi * (thigh_width_cm + estimated_depth) / 2
        
        return circumference
    
    def _estimate_bicep_circumference(self, landmarks: np.ndarray, scale_factor: float) -> float:
        """Estimate bicep circumference"""
        shoulder = landmarks[12]  # Right shoulder
        elbow = landmarks[14]     # Right elbow
        
        # Estimate upper arm width
        arm_length = euclidean([shoulder[0], shoulder[1]], [elbow[0], elbow[1]])
        
        # Bicep circumference is typically related to arm length
        # Empirical relationship for average build
        bicep_circumference = arm_length * scale_factor * 0.4
        
        return bicep_circumference
    
    def _calculate_confidence_scores(self, landmarks: np.ndarray, 
                                   segmentation: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate confidence scores for each measurement"""
        
        confidence_scores = {}
        
        # Base confidence on landmark visibility
        visibility_scores = landmarks[:, 3]  # Visibility is 4th column
        
        # Shoulder width confidence
        shoulder_visibility = np.mean([landmarks[11][3], landmarks[12][3]])
        confidence_scores['shoulder_width'] = min(shoulder_visibility, 1.0)
        
        # Arm length confidence
        arm_visibility = np.mean([landmarks[12][3], landmarks[14][3], landmarks[16][3]])
        confidence_scores['arm_length'] = min(arm_visibility, 1.0)
        
        # Hip measurements confidence
        hip_visibility = np.mean([landmarks[23][3], landmarks[24][3]])
        confidence_scores['hip_circumference'] = min(hip_visibility, 1.0)
        confidence_scores['waist_circumference'] = min(hip_visibility * 0.8, 1.0)  # Lower confidence for estimated waist
        
        # Leg measurements confidence
        leg_visibility = np.mean([landmarks[24][3], landmarks[26][3], landmarks[28][3]])
        confidence_scores['inseam_length'] = min(leg_visibility, 1.0)
        confidence_scores['thigh_circumference'] = min(leg_visibility * 0.7, 1.0)  # Lower for estimated circumference
        
        # Chest circumference (estimated from shoulders)
        confidence_scores['chest_circumference'] = min(shoulder_visibility * 0.6, 1.0)  # Lower for estimation
        
        # Bicep circumference (estimated)
        confidence_scores['bicep_circumference'] = min(arm_visibility * 0.5, 1.0)
        
        return confidence_scores
    
    def _combine_measurements(self, measurements_list: List[BodyMeasurements], 
                            weights: List[float]) -> BodyMeasurements:
        """Combine measurements from multiple views using weighted average"""
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        combined = BodyMeasurements()
        
        # Get all measurement fields
        measurement_fields = [field for field in asdict(measurements_list[0]).keys() 
                            if field != 'confidence_scores' and isinstance(getattr(measurements_list[0], field), (int, float))]
        
        # Calculate weighted averages
        for field in measurement_fields:
            values = [getattr(m, field) for m in measurements_list if getattr(m, field) > 0]
            if values:
                # Weight by corresponding confidence if available
                field_weights = []
                for i, m in enumerate(measurements_list):
                    if getattr(m, field) > 0:
                        confidence = m.confidence_scores.get(field, 1.0)
                        field_weights.append(weights[i] * confidence)
                
                if field_weights:
                    field_weights = np.array(field_weights)
                    field_weights = field_weights / np.sum(field_weights)
                    weighted_value = np.average(values, weights=field_weights)
                    setattr(combined, field, weighted_value)
        
        # Combine confidence scores
        combined.confidence_scores = {}
        for field in measurement_fields:
            confidences = [m.confidence_scores.get(field, 0.0) for m in measurements_list]
            if confidences:
                combined.confidence_scores[field] = np.average(confidences, weights=weights)
        
        return combined
    
    def _create_fallback_measurements(self, reference_measurement: Optional[Dict] = None) -> BodyMeasurements:
        """Create fallback measurements when MediaPipe is not available"""
        measurements = BodyMeasurements()
        
        # Use reference height if provided, otherwise use average
        if reference_measurement and reference_measurement.get('type') == 'height':
            measurements.height_cm = reference_measurement['value_cm']
        else:
            measurements.height_cm = 170.0  # Average height
        
        # Calculate other measurements based on anthropometric ratios
        measurements.shoulder_width = measurements.height_cm * 0.25
        measurements.chest_circumference = measurements.shoulder_width * 2.7
        measurements.waist_circumference = measurements.chest_circumference * 0.8
        measurements.hip_circumference = measurements.waist_circumference * 1.1
        measurements.arm_length = measurements.height_cm * 0.365
        measurements.torso_length = measurements.height_cm * 0.52
        measurements.inseam_length = measurements.height_cm * 0.48
        measurements.thigh_circumference = measurements.hip_circumference * 0.6
        measurements.bicep_circumference = measurements.arm_length * 0.4
        
        # Set low confidence scores for fallback measurements
        measurements.confidence_scores = {
            'height_cm': 0.3,
            'shoulder_width': 0.3,
            'chest_circumference': 0.3,
            'waist_circumference': 0.3,
            'hip_circumference': 0.3,
            'arm_length': 0.3,
            'torso_length': 0.3,
            'inseam_length': 0.3,
            'thigh_circumference': 0.3,
            'bicep_circumference': 0.3
        }
        
        return measurements


class MeasurementDatabase:
    """Database for storing and retrieving body measurements"""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
    
    def save_measurements(self, user_id: str, measurements: BodyMeasurements, 
                         measurement_method: str = "single_image") -> str:
        """Save measurements to database"""
        
        measurement_data = {
            'user_id': user_id,
            'measurements': asdict(measurements),
            'method': measurement_method,
            'timestamp': np.datetime64('now').isoformat(),
            'version': '1.0'
        }
        
        # In a real implementation, save to database
        # For now, save to JSON file
        measurement_id = f"{user_id}_{int(np.datetime64('now').astype('datetime64[s]').astype(int))}"
        
        os.makedirs("measurements", exist_ok=True)
        with open(f"measurements/{measurement_id}.json", 'w') as f:
            json.dump(measurement_data, f, indent=2)
        
        logger.info(f"Saved measurements for user {user_id}: {measurement_id}")
        return measurement_id
    
    def get_measurements(self, user_id: str) -> Optional[BodyMeasurements]:
        """Retrieve latest measurements for user"""
        
        # In a real implementation, query database
        # For now, load from JSON file
        try:
            import glob
            pattern = f"measurements/{user_id}_*.json"
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            # Get latest file
            latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            measurements_dict = data['measurements']
            measurements = BodyMeasurements(**measurements_dict)
            
            return measurements
            
        except Exception as e:
            logger.error(f"Failed to load measurements for user {user_id}: {e}")
            return None
    
    def get_measurements_for_garment(self, user_id: str, garment_type: str) -> Dict[str, float]:
        """Get relevant measurements for specific garment type"""
        
        measurements = self.get_measurements(user_id)
        if not measurements:
            return {}
        
        if garment_type not in GARMENT_MEASUREMENTS:
            logger.warning(f"Unknown garment type: {garment_type}")
            return {}
        
        required_measurements = GARMENT_MEASUREMENTS[garment_type]
        result = {}
        
        # Get primary measurements
        for field in required_measurements['primary']:
            value = getattr(measurements, field, 0.0)
            confidence = measurements.confidence_scores.get(field, 0.0)
            
            if value > 0:
                result[field] = {
                    'value': value,
                    'confidence': confidence,
                    'importance': 'primary'
                }
        
        # Get secondary measurements
        for field in required_measurements['secondary']:
            value = getattr(measurements, field, 0.0)
            confidence = measurements.confidence_scores.get(field, 0.0)
            
            if value > 0:
                result[field] = {
                    'value': value,
                    'confidence': confidence,
                    'importance': 'secondary'
                }
        
        # Get optional measurements
        for field in required_measurements.get('optional', []):
            value = getattr(measurements, field, 0.0)
            confidence = measurements.confidence_scores.get(field, 0.0)
            
            if value > 0:
                result[field] = {
                    'value': value,
                    'confidence': confidence,
                    'importance': 'optional'
                }
        
        return result


# Usage example
def example_measurement_extraction():
    """Example of how to use the measurement system"""
    
    # Initialize extractor
    extractor = AdvancedMeasurementExtractor()
    
    # Load image
    image = cv2.imread("user_photo.jpg")
    
    # Option 1: Single image with known height
    reference = {'type': 'height', 'value_cm': 170}
    measurements = extractor.extract_measurements_single_image(image, reference)
    
    # Option 2: Multiple images for better accuracy
    front_image = cv2.imread("front_view.jpg")
    side_image = cv2.imread("side_view.jpg")
    images = [front_image, side_image]
    
    measurements = extractor.extract_measurements_multi_view(images, reference)
    
    # Save to database
    db = MeasurementDatabase()
    measurement_id = db.save_measurements("user123", measurements, "multi_view")
    
    # Get measurements for specific garment
    shirt_measurements = db.get_measurements_for_garment("user123", "shirts")
    
    print("Measurements for shirt:", shirt_measurements)
    
    return measurements


if __name__ == "__main__":
    # Run example
    measurements = example_measurement_extraction()
    print("Extracted measurements:", asdict(measurements))
