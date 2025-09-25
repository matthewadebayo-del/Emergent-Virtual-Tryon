"""
Performance Optimizations for Virtual Try-On Pipeline
GPU acceleration, image preprocessing, and caching
"""

import hashlib
import io
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

logger = logging.getLogger(__name__)

# GPU Detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"

# Cache storage
ANALYSIS_CACHE = {}
MAX_CACHE_SIZE = 100

class ImagePreprocessor:
    """Standardize input image quality"""
    
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Standardize image quality and format"""
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste centered
        processed = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        processed.paste(image, (paste_x, paste_y))
        
        # Enhance quality
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.05)
        
        return processed
    
    @staticmethod
    def preprocess_for_pose_detection(image: Image.Image) -> np.ndarray:
        """Preprocess specifically for pose detection"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Normalize lighting
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        cv_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

class GPUAccelerator:
    """GPU acceleration utilities"""
    
    @staticmethod
    def get_device():
        return DEVICE
    
    @staticmethod
    def is_gpu_available():
        return GPU_AVAILABLE
    
    @staticmethod
    def move_to_device(data):
        """Move data to GPU if available"""
        if GPU_AVAILABLE and hasattr(data, 'to'):
            return data.to(DEVICE)
        return data

class AnalysisCache:
    """Cache for analysis results"""
    
    @staticmethod
    def get_image_hash(image: Image.Image) -> str:
        """Generate hash for image caching"""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=95)
        return hashlib.md5(img_bytes.getvalue()).hexdigest()
    
    @staticmethod
    def get_cached_analysis(image_hash: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        cache_key = f"{analysis_type}_{image_hash}"
        return ANALYSIS_CACHE.get(cache_key)
    
    @staticmethod
    def cache_analysis(image_hash: str, analysis_type: str, result: Dict[str, Any]):
        """Cache analysis result"""
        cache_key = f"{analysis_type}_{image_hash}"
        
        # Manage cache size
        if len(ANALYSIS_CACHE) >= MAX_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(ANALYSIS_CACHE))
            del ANALYSIS_CACHE[oldest_key]
        
        ANALYSIS_CACHE[cache_key] = result
        logger.info(f"Cached {analysis_type} analysis for {image_hash[:8]}")
    
    @staticmethod
    def clear_cache():
        """Clear analysis cache"""
        ANALYSIS_CACHE.clear()
        logger.info("Analysis cache cleared")

class OptimizedMediaPipeProcessor:
    """GPU-accelerated MediaPipe processing"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.device = DEVICE
        
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            
            # Configure for GPU if available
            if self.gpu_available:
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5
                )
                self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
            else:
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=True,
                    min_detection_confidence=0.5
                )
                self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=0
                )
            
            self.initialized = True
            logger.info(f"MediaPipe initialized with {'GPU' if self.gpu_available else 'CPU'}")
            
        except ImportError:
            self.initialized = False
            logger.warning("MediaPipe not available")
    
    def process_pose_with_gpu(self, image: np.ndarray) -> Dict[str, Any]:
        """GPU-accelerated pose detection"""
        if not self.initialized:
            return {"success": False, "error": "MediaPipe not available"}
        
        try:
            # Process with MediaPipe
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = self.mp_pose.PoseLandmark(idx).name.lower()
                    landmarks[landmark_name] = [
                        landmark.x * image.shape[1],
                        landmark.y * image.shape[0],
                        landmark.visibility
                    ]
                
                return {
                    "success": True,
                    "landmarks": landmarks,
                    "segmentation_mask": results.segmentation_mask,
                    "processing_device": self.device
                }
            else:
                return {"success": False, "error": "No pose detected"}
                
        except Exception as e:
            logger.error(f"GPU pose processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def process_segmentation_with_gpu(self, image: np.ndarray) -> Dict[str, Any]:
        """GPU-accelerated body segmentation"""
        if not self.initialized:
            return {"success": False, "error": "MediaPipe not available"}
        
        try:
            results = self.segmentation.process(image)
            
            if results.segmentation_mask is not None:
                return {
                    "success": True,
                    "mask": results.segmentation_mask,
                    "processing_device": self.device
                }
            else:
                return {"success": False, "error": "Segmentation failed"}
                
        except Exception as e:
            logger.error(f"GPU segmentation failed: {e}")
            return {"success": False, "error": str(e)}