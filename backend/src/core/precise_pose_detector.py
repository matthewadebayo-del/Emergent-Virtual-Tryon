import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("[WARN] MediaPipe not available in PrecisePoseDetector - using fallback mode")

class PrecisePoseDetector:
    """
    High-precision pose detection using MediaPipe with optimized settings
    for virtual try-on applications.
    """
    
    def __init__(self, 
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 smooth_landmarks: bool = True):
        """
        Initialize the pose detector with optimized settings.
        
        Args:
            model_complexity: 0=Lite, 1=Full, 2=Heavy (most accurate)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            smooth_landmarks: Enable landmark smoothing for video
        """
        self.initialized = MEDIAPIPE_AVAILABLE
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Initialize pose detection with optimal settings for try-on
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,  # Set to True for single images
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                enable_segmentation=True,  # Useful for background removal
                smooth_segmentation=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            self.pose = None
        
        # Key landmarks for virtual try-on
        self.key_landmarks = {
            'shoulders': [11, 12],  # Left and right shoulders
            'elbows': [13, 14],     # Left and right elbows
            'wrists': [15, 16],     # Left and right wrists
            'hips': [23, 24],       # Left and right hips
            'knees': [25, 26],      # Left and right knees
            'ankles': [27, 28],     # Left and right ankles
            'nose': [0],            # Face reference
            'chest': [11, 12],      # For chest measurements
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_pose(self, image: np.ndarray) -> Dict:
        """
        Detect pose landmarks in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing landmarks, confidence scores, and segmentation mask
        """
        if not self.initialized:
            return {
                'landmarks': None,
                'world_landmarks': None,
                'segmentation_mask': None,
                'confidence': 0.0,
                'pose_present': False
            }
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        return {
            'landmarks': results.pose_landmarks,
            'world_landmarks': results.pose_world_landmarks,
            'segmentation_mask': results.segmentation_mask,
            'confidence': self._calculate_overall_confidence(results.pose_landmarks),
            'pose_present': results.pose_landmarks is not None
        }
    
    def extract_key_points(self, landmarks) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract key body points needed for virtual try-on.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of key body points with normalized coordinates
        """
        if not landmarks:
            return {}
        
        key_points = {}
        
        for body_part, indices in self.key_landmarks.items():
            points = []
            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    points.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
            key_points[body_part] = points
        
        return key_points
    
    def calculate_body_measurements(self, landmarks, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate body measurements from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_shape: (height, width) of the image
            
        Returns:
            Dictionary of body measurements in pixels
        """
        if not landmarks:
            return {}
        
        height, width = image_shape[:2]
        measurements = {}
        
        # Convert normalized coordinates to pixel coordinates
        def to_pixels(landmark):
            return (int(landmark.x * width), int(landmark.y * height))
        
        try:
            # Shoulder width
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                ls_px = to_pixels(left_shoulder)
                rs_px = to_pixels(right_shoulder)
                measurements['shoulder_width'] = np.linalg.norm(np.array(ls_px) - np.array(rs_px))
            
            # Torso length (shoulder to hip)
            if left_shoulder.visibility > 0.5:
                left_hip = landmarks.landmark[23]
                if left_hip.visibility > 0.5:
                    ls_px = to_pixels(left_shoulder)
                    lh_px = to_pixels(left_hip)
                    measurements['torso_length'] = np.linalg.norm(np.array(ls_px) - np.array(lh_px))
            
            # Arm length (shoulder to wrist)
            left_wrist = landmarks.landmark[15]
            if left_shoulder.visibility > 0.5 and left_wrist.visibility > 0.5:
                ls_px = to_pixels(left_shoulder)
                lw_px = to_pixels(left_wrist)
                measurements['arm_length'] = np.linalg.norm(np.array(ls_px) - np.array(lw_px))
            
            # Hip width
            left_hip = landmarks.landmark[23]
            right_hip = landmarks.landmark[24]
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                lh_px = to_pixels(left_hip)
                rh_px = to_pixels(right_hip)
                measurements['hip_width'] = np.linalg.norm(np.array(lh_px) - np.array(rh_px))
            
            # Leg length (hip to ankle)
            left_ankle = landmarks.landmark[27]
            if left_hip.visibility > 0.5 and left_ankle.visibility > 0.5:
                lh_px = to_pixels(left_hip)
                la_px = to_pixels(left_ankle)
                measurements['leg_length'] = np.linalg.norm(np.array(lh_px) - np.array(la_px))
                
        except Exception as e:
            self.logger.warning(f"Error calculating measurements: {e}")
        
        return measurements
    
    def draw_pose_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw pose landmarks on the image.
        
        Args:
            image: Input image
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Image with drawn landmarks
        """
        annotated_image = image.copy()
        
        if landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return annotated_image
    
    def _calculate_overall_confidence(self, landmarks) -> float:
        """Calculate overall confidence based on key landmark visibility."""
        if not landmarks:
            return 0.0
        
        key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Key points for try-on
        total_confidence = 0.0
        count = 0
        
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                total_confidence += landmarks.landmark[idx].visibility
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def is_pose_suitable_for_tryon(self, landmarks, min_confidence: float = 0.3) -> Tuple[bool, str]:
        """
        Check if the detected pose is suitable for virtual try-on.
        
        Args:
            landmarks: MediaPipe pose landmarks
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (is_suitable: bool, reason: str)
        """
        if not landmarks:
            return False, "No pose detected"
        
        confidence = self._calculate_overall_confidence(landmarks)
        if confidence < min_confidence:
            return False, f"Low confidence: {confidence:.2f}"
        
        # Check if key landmarks are visible
        key_landmarks = [11, 12, 23, 24]  # Shoulders and hips
        for idx in key_landmarks:
            if idx < len(landmarks.landmark):
                if landmarks.landmark[idx].visibility < 0.3:
                    return False, f"Key landmark {idx} not visible"
            else:
                return False, f"Missing landmark {idx}"
        
        return True, "Pose suitable for try-on"
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()