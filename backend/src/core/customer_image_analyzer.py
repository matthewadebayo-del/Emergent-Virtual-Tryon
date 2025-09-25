"""
Enhanced Customer Image Analysis Pipeline
Implements comprehensive computer vision analysis for customer photos
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import io
from .performance_optimizations import OptimizedMediaPipeProcessor, ImagePreprocessor, GPUAccelerator

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class CustomerImageAnalyzer:
    """Enhanced customer image analysis with pose detection, measurements, and skin tone analysis"""
    
    def __init__(self):
        self.initialized = MEDIAPIPE_AVAILABLE
        self.gpu_processor = OptimizedMediaPipeProcessor()
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.7
            )
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.7
            )
        else:
            self.mp_pose = None
            self.mp_face = None
    
    def analyze_customer_image_optimized(self, image_input, reference_height_cm: Optional[float] = None) -> Dict[str, Any]:
        """GPU-accelerated customer image analysis with preprocessing"""
        try:
            # Convert input to PIL Image
            if isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return {"analysis_success": False, "error": "Invalid image input"}
            
            # Preprocess for optimal pose detection
            preprocessed_array = ImagePreprocessor.preprocess_for_pose_detection(image)
            
            # GPU-accelerated pose detection
            pose_result = self.gpu_processor.process_pose_with_gpu(preprocessed_array)
            
            if not pose_result["success"]:
                return self._fallback_analysis(image, reference_height_cm)
            
            # Extract measurements from GPU results
            landmarks = pose_result["landmarks"]
            measurements = self._calculate_measurements_from_landmarks(landmarks, reference_height_cm)
            
            # GPU-accelerated segmentation
            segmentation_result = self.gpu_processor.process_segmentation_with_gpu(preprocessed_array)
            
            # Skin tone analysis
            skin_tone = self._analyze_skin_tone_optimized(image, segmentation_result.get("mask"))
            
            return {
                "analysis_success": True,
                "pose_keypoints": self._convert_landmarks_to_keypoints(landmarks),
                "pose_landmarks": landmarks,
                "measurements": measurements,
                "body_segmentation": segmentation_result,
                "skin_tone": skin_tone,
                "confidence_score": self._calculate_confidence_score(landmarks),
                "processing_device": GPUAccelerator.get_device(),
                "gpu_accelerated": GPUAccelerator.is_gpu_available()
            }
            
        except Exception as e:
            print(f"[CUSTOMER] GPU analysis failed: {e}, using fallback")
            return self._fallback_analysis(image, reference_height_cm)
    
    def analyze_customer_image(self, image_bytes: bytes, reference_height_cm: Optional[float] = None) -> Dict[str, Any]:
        """Main analysis function - comprehensive customer image analysis"""
        try:
            # Convert bytes to image array
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Run all analysis components
            pose_data = self._detect_pose(image)
            measurements = self._extract_measurements(pose_data, image.shape[:2], reference_height_cm)
            body_mask = self._segment_body(image, pose_data.get("segmentation_mask"))
            skin_tone = self._detect_skin_tone(image, pose_data.get("landmarks"))
            scale_info = self._calculate_scale(pose_data, measurements, image.shape[:2])
            
            result = {
                "pose_landmarks": pose_data.get("landmarks"),
                "pose_keypoints": self._convert_landmarks_to_keypoints_dict(pose_data.get("landmarks")),
                "measurements": measurements,
                "body_segmentation": body_mask,
                "skin_tone": skin_tone,
                "scale_info": scale_info,
                "image_shape": image.shape,
                "analysis_success": True,
                "confidence_score": measurements.get("overall_confidence", 0.8)
            }
            
            print(f"[CUSTOMER] Analysis result keys: {list(result.keys())}")
            print(f"[CUSTOMER] Pose keypoints: {result['pose_keypoints'] is not None}")
            print(f"[CUSTOMER] Measurements: {result['measurements']}")
            
            return result
            
        except Exception as e:
            print(f"[CUSTOMER] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(image_bytes)
    
    def _fallback_analysis(self, image: Image.Image, reference_height_cm: Optional[float] = None) -> Dict[str, Any]:
        """Fallback analysis when GPU processing fails"""
        # Convert PIL to bytes for original method
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return self.analyze_customer_image(img_bytes.getvalue(), reference_height_cm)
    
    def _calculate_measurements_from_landmarks(self, landmarks: Dict[str, Any], reference_height_cm: Optional[float] = None) -> Dict[str, Any]:
        """Calculate measurements from GPU-processed landmarks"""
        try:
            # Extract key points
            left_shoulder = landmarks.get("left_shoulder", [0, 0, 0])
            right_shoulder = landmarks.get("right_shoulder", [0, 0, 0])
            nose = landmarks.get("nose", [0, 0, 0])
            left_ankle = landmarks.get("left_ankle", [0, 0, 0])
            right_ankle = landmarks.get("right_ankle", [0, 0, 0])
            
            # Calculate shoulder width
            shoulder_width_px = abs(left_shoulder[0] - right_shoulder[0]) if left_shoulder[2] > 0.7 and right_shoulder[2] > 0.7 else 0
            
            # Calculate height (nose to best ankle)
            height_px = 0
            if nose[2] > 0.7:
                if left_ankle[2] > right_ankle[2] and left_ankle[2] > 0.7:
                    height_px = abs(nose[1] - left_ankle[1])
                elif right_ankle[2] > 0.7:
                    height_px = abs(nose[1] - right_ankle[1])
            
            # Calculate scale factor
            if reference_height_cm and height_px > 0:
                scale_factor = reference_height_cm / height_px
            elif shoulder_width_px > 0:
                scale_factor = 45.0 / shoulder_width_px
            else:
                scale_factor = 170.0 / max(height_px, 1000)
            
            return {
                "shoulder_width_cm": shoulder_width_px * scale_factor if shoulder_width_px > 0 else 45.0,
                "height_cm": height_px * scale_factor if height_px > 0 else 170.0,
                "scale_factor": scale_factor,
                "overall_confidence": min(left_shoulder[2], right_shoulder[2]) if shoulder_width_px > 0 else 0.7
            }
            
        except Exception as e:
            print(f"[CUSTOMER] Measurement calculation failed: {e}")
            return self._create_fallback_measurements()
    
    def _convert_landmarks_to_keypoints(self, landmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Convert GPU landmarks to keypoints format"""
        keypoints = {}
        for name, coords in landmarks.items():
            if len(coords) >= 2:
                keypoints[name] = [coords[0], coords[1]]
        return keypoints
    
    def _calculate_confidence_score(self, landmarks: Dict[str, Any]) -> float:
        """Calculate overall confidence from landmarks"""
        confidences = [coords[2] for coords in landmarks.values() if len(coords) > 2]
        return np.mean(confidences) if confidences else 0.7
    
    def _analyze_skin_tone_optimized(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """GPU-optimized skin tone analysis"""
        try:
            img_array = np.array(image)
            
            if mask is not None:
                # Use GPU-generated segmentation mask
                mask_binary = (mask > 0.5).astype(np.uint8)
                face_region = img_array[mask_binary == 1]
            else:
                # Fallback to center region
                h, w = img_array.shape[:2]
                face_region = img_array[h//4:3*h//4, w//4:3*w//4].reshape(-1, 3)
            
            if len(face_region) == 0:
                return {"rgb": [120, 100, 80], "warmth": "neutral", "confidence": 0.3}
            
            # Fast skin tone calculation
            avg_color = np.mean(face_region, axis=0).astype(int)
            
            # Determine warmth
            r, g, b = avg_color
            if r > g and (r - b) > 10:
                warmth = "warm"
            elif b > r and (b - r) > 10:
                warmth = "cool"
            else:
                warmth = "neutral"
            
            return {
                "rgb": avg_color.tolist(),
                "warmth": warmth,
                "confidence": 0.8 if mask is not None else 0.6
            }
            
        except Exception as e:
            print(f"[CUSTOMER] Skin tone analysis failed: {e}")
            return {"rgb": [120, 100, 80], "warmth": "neutral", "confidence": 0.3}
    
    def _detect_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect 13+ body keypoints using MediaPipe"""
        if not self.initialized:
            return {"landmarks": None, "segmentation_mask": None}
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return {"landmarks": None, "segmentation_mask": None}
        
        # Extract landmarks as numpy array
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return {
            "landmarks": np.array(landmarks),
            "segmentation_mask": results.segmentation_mask,
            "pose_detected": True
        }
    
    def _extract_measurements(self, pose_data: Dict[str, Any], image_shape: Tuple[int, int], reference_height_cm: Optional[float] = None) -> Dict[str, Any]:
        """Calculate real-world measurements from pose keypoints"""
        landmarks = pose_data.get("landmarks")
        
        if landmarks is None:
            return self._create_fallback_measurements()
        
        h, w = image_shape[:2]
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= w
        pixel_landmarks[:, 1] *= h
        
        # MediaPipe pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        NOSE = 0
        
        measurements = {}
        confidence_scores = {}
        
        # Shoulder width
        if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[RIGHT_SHOULDER][3] > 0.7:
            shoulder_width_px = abs(pixel_landmarks[LEFT_SHOULDER][0] - pixel_landmarks[RIGHT_SHOULDER][0])
            measurements["shoulder_width_px"] = shoulder_width_px
            confidence_scores["shoulder_width"] = min(landmarks[LEFT_SHOULDER][3], landmarks[RIGHT_SHOULDER][3])
        else:
            measurements["shoulder_width_px"] = 0
            confidence_scores["shoulder_width"] = 0.0
        
        # Body height (nose to ankle)
        best_ankle_confidence = 0
        height_px = 0
        if landmarks[NOSE][3] > 0.7:
            if landmarks[LEFT_ANKLE][3] > landmarks[RIGHT_ANKLE][3]:
                if landmarks[LEFT_ANKLE][3] > 0.7:
                    height_px = abs(pixel_landmarks[NOSE][1] - pixel_landmarks[LEFT_ANKLE][1])
                    best_ankle_confidence = landmarks[LEFT_ANKLE][3]
            else:
                if landmarks[RIGHT_ANKLE][3] > 0.7:
                    height_px = abs(pixel_landmarks[NOSE][1] - pixel_landmarks[RIGHT_ANKLE][1])
                    best_ankle_confidence = landmarks[RIGHT_ANKLE][3]
        
        measurements["height_px"] = height_px
        confidence_scores["height"] = min(landmarks[NOSE][3], best_ankle_confidence) if height_px > 0 else 0.0
        
        # Calculate scale (pixels per cm)
        scale_factor = self._calculate_scale_factor(measurements, reference_height_cm)
        
        # Calculate additional measurements from landmarks
        additional_measurements = self._calculate_additional_measurements(landmarks, pixel_landmarks, scale_factor, confidence_scores)
        
        # Convert to real-world measurements with all body measurements
        real_measurements = {
            "shoulder_width_cm": measurements["shoulder_width_px"] * scale_factor if measurements["shoulder_width_px"] > 0 else 45.0,
            "height_cm": measurements["height_px"] * scale_factor if measurements["height_px"] > 0 else 170.0,
            "scale_factor": scale_factor,
            "confidence_scores": confidence_scores,
            "overall_confidence": np.mean(list(confidence_scores.values())) if confidence_scores else 0.7
        }
        
        # Add all derived measurements
        real_measurements.update(additional_measurements)
        
        print(f"[CUSTOMER] Derived measurements: {list(real_measurements.keys())}")
        print(f"[CUSTOMER] Chest: {real_measurements.get('chest', 'NOT_CALCULATED')}cm")
        
        return real_measurements
        
        return real_measurements
    
    def _calculate_additional_measurements(self, landmarks: np.ndarray, pixel_landmarks: np.ndarray, scale_factor: float, confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate chest, waist, hips and other measurements from pose landmarks"""
        try:
            # MediaPipe pose landmark indices
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            
            additional_measurements = {}
            
            # Chest width (estimate from shoulder width with anatomical ratio)
            shoulder_width_px = abs(pixel_landmarks[LEFT_SHOULDER][0] - pixel_landmarks[RIGHT_SHOULDER][0]) if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[RIGHT_SHOULDER][3] > 0.7 else 0
            if shoulder_width_px > 0:
                # Chest is typically 1.8-2.2x shoulder width
                chest_circumference_cm = (shoulder_width_px * scale_factor) * 2.0
                additional_measurements["chest"] = chest_circumference_cm
                additional_measurements["chest_circumference_cm"] = chest_circumference_cm
                confidence_scores["chest"] = min(landmarks[LEFT_SHOULDER][3], landmarks[RIGHT_SHOULDER][3])
            else:
                additional_measurements["chest"] = 90.0  # Default
                additional_measurements["chest_circumference_cm"] = 90.0
                confidence_scores["chest"] = 0.3
            
            # Hip width and circumference
            if landmarks[LEFT_HIP][3] > 0.7 and landmarks[RIGHT_HIP][3] > 0.7:
                hip_width_px = abs(pixel_landmarks[LEFT_HIP][0] - pixel_landmarks[RIGHT_HIP][0])
                hip_circumference_cm = (hip_width_px * scale_factor) * 2.2  # Hips are typically wider than shoulders
                additional_measurements["hips"] = hip_circumference_cm
                additional_measurements["hip_circumference_cm"] = hip_circumference_cm
                confidence_scores["hips"] = min(landmarks[LEFT_HIP][3], landmarks[RIGHT_HIP][3])
            else:
                additional_measurements["hips"] = 95.0  # Default
                additional_measurements["hip_circumference_cm"] = 95.0
                confidence_scores["hips"] = 0.3
            
            # Waist (estimate as midpoint between chest and hips)
            chest_val = additional_measurements.get("chest", 90.0)
            hips_val = additional_measurements.get("hips", 95.0)
            waist_circumference_cm = (chest_val + hips_val) / 2.5  # Waist is typically smaller
            additional_measurements["waist"] = waist_circumference_cm
            additional_measurements["waist_circumference_cm"] = waist_circumference_cm
            confidence_scores["waist"] = (confidence_scores.get("chest", 0.3) + confidence_scores.get("hips", 0.3)) / 2
            
            # Arm length (shoulder to wrist)
            if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[LEFT_ELBOW][3] > 0.7:
                arm_length_px = np.sqrt(
                    (pixel_landmarks[LEFT_SHOULDER][0] - pixel_landmarks[LEFT_ELBOW][0])**2 +
                    (pixel_landmarks[LEFT_SHOULDER][1] - pixel_landmarks[LEFT_ELBOW][1])**2
                )
                additional_measurements["arm_length"] = arm_length_px * scale_factor
                confidence_scores["arm_length"] = min(landmarks[LEFT_SHOULDER][3], landmarks[LEFT_ELBOW][3])
            else:
                additional_measurements["arm_length"] = 60.0  # Default
                confidence_scores["arm_length"] = 0.3
            
            # Torso length (shoulder to hip)
            if landmarks[LEFT_SHOULDER][3] > 0.7 and landmarks[LEFT_HIP][3] > 0.7:
                torso_length_px = abs(pixel_landmarks[LEFT_SHOULDER][1] - pixel_landmarks[LEFT_HIP][1])
                additional_measurements["torso_length"] = torso_length_px * scale_factor
                confidence_scores["torso_length"] = min(landmarks[LEFT_SHOULDER][3], landmarks[LEFT_HIP][3])
            else:
                additional_measurements["torso_length"] = 60.0  # Default
                confidence_scores["torso_length"] = 0.3
            
            print(f"[CUSTOMER] Calculated additional measurements: chest={additional_measurements.get('chest', 0):.1f}cm, waist={additional_measurements.get('waist', 0):.1f}cm, hips={additional_measurements.get('hips', 0):.1f}cm")
            
            return additional_measurements
            
        except Exception as e:
            print(f"[CUSTOMER] Additional measurements calculation failed: {e}")
            return {
                "chest": 90.0,
                "chest_circumference_cm": 90.0,
                "waist": 75.0,
                "waist_circumference_cm": 75.0,
                "hips": 95.0,
                "hip_circumference_cm": 95.0,
                "arm_length": 60.0,
                "torso_length": 60.0
            }
    
    def _calculate_scale_factor(self, measurements: Dict[str, float], reference_height_cm: Optional[float] = None) -> float:
        """Calculate pixels per cm ratio"""
        if reference_height_cm and measurements["height_px"] > 0:
            return reference_height_cm / measurements["height_px"]
        elif measurements["shoulder_width_px"] > 0:
            return 45.0 / measurements["shoulder_width_px"]
        else:
            return 170.0 / max(measurements.get("height_px", 1000), 1000)
    
    def _segment_body(self, image: np.ndarray, segmentation_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """Separate person from background"""
        if segmentation_mask is not None:
            mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
            return {"mask": mask, "method": "mediapipe_segmentation", "success": True}
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return {"mask": mask, "method": "threshold_segmentation", "success": False}
    
    def _detect_skin_tone(self, image: np.ndarray, landmarks: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze face/hand regions for dominant skin color"""
        if landmarks is None or not SKLEARN_AVAILABLE:
            return self._fallback_skin_tone()
        
        h, w = image.shape[:2]
        NOSE = 0
        if landmarks[NOSE][3] > 0.7:
            nose_x = int(landmarks[NOSE][0] * w)
            nose_y = int(landmarks[NOSE][1] * h)
            
            face_size = min(w, h) // 8
            x1 = max(0, nose_x - face_size)
            y1 = max(0, nose_y - face_size)
            x2 = min(w, nose_x + face_size)
            y2 = min(h, nose_y + face_size)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size > 0:
                return self._analyze_skin_color(face_region)
        
        return self._fallback_skin_tone()
    
    def _analyze_skin_color(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Extract dominant skin color from face region"""
        try:
            pixels = face_region.reshape(-1, 3)
            mask = np.all((pixels > 50) & (pixels < 220), axis=1)
            skin_pixels = pixels[mask]
            
            if len(skin_pixels) < 10:
                return self._fallback_skin_tone()
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(skin_pixels)
            
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)
            
            rgb_color = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
            brightness = np.mean(dominant_color)
            category = "dark" if brightness < 100 else "medium" if brightness < 160 else "light"
            
            return {
                "rgb_color": rgb_color,
                "hex_color": "#{:02x}{:02x}{:02x}".format(*rgb_color),
                "category": category,
                "brightness": float(brightness),
                "analysis_success": True
            }
            
        except Exception as e:
            print(f"[SKIN] Skin tone analysis failed: {e}")
            return self._fallback_skin_tone()
    
    def _calculate_scale(self, pose_data: Dict[str, Any], measurements: Dict[str, Any], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate scale information for measurements"""
        return {
            "pixels_per_cm": measurements.get("scale_factor", 1.0),
            "image_width": image_shape[1],
            "image_height": image_shape[0],
            "reference_used": "shoulder_width" if measurements.get("shoulder_width_px", 0) > 0 else "height",
            "scale_confidence": measurements.get("confidence_scores", {}).get("shoulder_width", 0.7)
        }
    
    def _create_fallback_analysis(self, image_bytes: bytes) -> Dict[str, Any]:
        """Create fallback analysis when pose detection fails"""
        result = {
            "pose_landmarks": None,
            "pose_keypoints": None,
            "measurements": self._create_fallback_measurements(),
            "body_segmentation": {"mask": None, "method": "none", "success": False},
            "skin_tone": self._fallback_skin_tone(),
            "scale_info": {"pixels_per_cm": 1.0, "reference_used": "fallback"},
            "image_shape": (512, 512, 3),
            "analysis_success": False,
            "confidence_score": 0.3
        }
        
        print(f"[CUSTOMER] Using fallback analysis - pose_keypoints: {result['pose_keypoints']}")
        return result
    
    def _create_fallback_measurements(self) -> Dict[str, Any]:
        """Create fallback measurements when pose detection fails"""
        return {
            "shoulder_width_cm": 45.0,
            "height_cm": 170.0,
            "confidence_scores": {},
            "overall_confidence": 0.3
        }
    
    def _convert_landmarks_to_keypoints_dict(self, landmarks: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Convert MediaPipe landmarks array to keypoints dictionary"""
        if landmarks is None:
            return None
        
        try:
            # MediaPipe pose landmark indices
            keypoint_mapping = {
                "nose": 0,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_hip": 23,
                "right_hip": 24,
                "left_wrist": 15,
                "right_wrist": 16,
                "left_ankle": 27,
                "right_ankle": 28
            }
            
            keypoints = {}
            for name, idx in keypoint_mapping.items():
                if idx < len(landmarks) and landmarks[idx][3] > 0.5:  # visibility threshold
                    keypoints[name] = [landmarks[idx][0], landmarks[idx][1]]
            
            return keypoints if keypoints else None
            
        except Exception as e:
            print(f"[CUSTOMER] Keypoint conversion failed: {e}")
            return None
    
    def _fallback_skin_tone(self) -> Dict[str, Any]:
        """Fallback skin tone when detection fails"""
        return {
            "rgb_color": (200, 180, 160),
            "hex_color": "#c8b4a0",
            "category": "medium",
            "brightness": 180.0,
            "analysis_success": False
        }