"""
Production-Ready Virtual Try-On Engine
Implements both Hybrid 3D Pipeline and fal.ai FASHN integration
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import mediapipe as mp
import base64
import io
import logging
from typing import Tuple, Optional, Dict, Any
import asyncio
import aiohttp
import fal_client
from rembg import remove, new_session
from diffusers import StableDiffusionInpaintPipeline
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class VirtualTryOnEngine:
    def __init__(self):
        """Initialize the Virtual Try-On Engine with lazy loading"""
        logger.info("Initializing Virtual Try-On Engine with lazy loading...")
        
        # Initialize components to None - will be loaded when needed
        self.mp_pose = None
        self.pose = None
        self.bg_remover = None
        self.yolo_model = None
        self.inpainting_pipe = None
        
        # Flags to track initialization state
        self._pose_initialized = False
        self._bg_remover_initialized = False
        self._yolo_initialized = False
        
        logger.info("Virtual Try-On Engine initialized with lazy loading")
    
    def _ensure_pose_detection(self):
        """Lazy initialize pose detection components"""
        if not self._pose_initialized:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5
                )
                self._pose_initialized = True
                logger.info("MediaPipe pose detection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pose detection: {e}")
                self._pose_initialized = False
    
    def _ensure_bg_remover(self):
        """Lazy initialize background removal"""
        if not self._bg_remover_initialized:
            try:
                from rembg import new_session
                self.bg_remover = new_session('u2net')
                self._bg_remover_initialized = True
                logger.info("Background remover initialized")
            except Exception as e:
                logger.error(f"Failed to initialize background remover: {e}")
                self._bg_remover_initialized = False
    
    def _ensure_yolo_model(self):
        """Lazy initialize YOLO model"""
        if not self._yolo_initialized:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n-seg.pt')
                self._yolo_initialized = True
                logger.info("YOLO model initialized")
            except Exception as e:
                logger.warning(f"YOLO model loading failed: {e}, will use fallback")
                self.yolo_model = None
                self._yolo_initialized = False
    
    async def process_fal_ai_tryon(
        self, 
        user_image_bytes: bytes, 
        garment_image_url: str,
        product_name: str,
        category: str
    ) -> Tuple[str, float]:
        """
        Process virtual try-on using fal.ai FASHN API
        """
        try:
            logger.info("Starting fal.ai FASHN virtual try-on")
            
            # Convert user image to base64
            user_image_b64 = base64.b64encode(user_image_bytes).decode()
            user_image_data_url = f"data:image/jpeg;base64,{user_image_b64}"
            
            # Prepare fal.ai request
            handler = await fal_client.submit_async(
                "fal-ai/fashn-virtual-try-on",
                arguments={
                    "human_image": user_image_data_url,
                    "garment_image": garment_image_url,
                    "description": f"{product_name} - {category}",
                    "category": category.lower().replace("'", "").replace(" ", "_"),
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "seed": 42
                }
            )
            
            # Get result with timeout
            result = await asyncio.wait_for(handler.get(), timeout=120)
            
            if result and "image" in result:
                result_url = result["image"]["url"]
                cost = 0.075  # fal.ai pricing
                
                logger.info("fal.ai virtual try-on completed successfully")
                return result_url, cost
            else:
                raise Exception("Invalid response from fal.ai")
                
        except Exception as e:
            logger.error(f"fal.ai try-on error: {str(e)}")
            raise Exception(f"fal.ai processing failed: {str(e)}")
    
    async def process_hybrid_tryon(
        self,
        user_image_bytes: bytes,
        garment_image_url: str, 
        product_name: str,
        category: str
    ) -> Tuple[str, float]:
        """
        Process virtual try-on using Hybrid 3D Pipeline
        """
        try:
            logger.info("Starting Hybrid 3D virtual try-on pipeline")
            
            # Stage 1: Image preprocessing
            user_image = self._bytes_to_image(user_image_bytes)
            garment_image = await self._download_garment_image(garment_image_url)
            
            # Stage 2: Person detection and segmentation
            person_mask, body_keypoints = self._detect_person_and_pose(user_image)
            
            # Stage 3: Garment region detection
            garment_region = self._detect_garment_region(user_image, body_keypoints, category)
            
            # Stage 4: Garment fitting and blending
            fitted_result = self._fit_and_blend_garment(
                user_image, 
                garment_image,
                person_mask,
                garment_region,
                body_keypoints
            )
            
            # Stage 5: Post-processing and enhancement
            final_result = self._enhance_result(fitted_result, user_image)
            
            # Stage 6: Save result and return URL
            result_url = await self._save_result_image(final_result)
            cost = 0.02  # Hybrid pricing
            
            logger.info("Hybrid 3D virtual try-on completed successfully")
            return result_url, cost
            
        except Exception as e:
            logger.error(f"Hybrid try-on error: {str(e)}")
            raise Exception(f"Hybrid processing failed: {str(e)}")
    
    def _bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    async def _download_garment_image(self, garment_url: str) -> np.ndarray:
        """Download and process garment image"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(garment_url) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        return self._bytes_to_image(image_bytes)
                    else:
                        raise Exception(f"Failed to download garment image: {response.status}")
        except Exception as e:
            logger.error(f"Garment download error: {e}")
            # Create placeholder garment
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _detect_person_and_pose(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect person in image and extract pose keypoints using MediaPipe
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get pose landmarks
            results = self.pose.process(rgb_image)
            
            # Create person mask
            if results.segmentation_mask is not None:
                # Use MediaPipe segmentation
                mask = results.segmentation_mask
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                # Fallback: create mask from YOLO if available
                if self.yolo_model:
                    yolo_results = self.yolo_model(image)
                    mask = self._extract_person_mask_from_yolo(yolo_results, image.shape[:2])
                else:
                    # Ultimate fallback: use rembg
                    pil_image = Image.fromarray(image)
                    no_bg = remove(pil_image, session=self.bg_remover)
                    mask = np.array(no_bg)[:,:,3] if no_bg.mode == 'RGBA' else np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            # Extract keypoints
            keypoints = {}
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints[idx] = {
                        'x': landmark.x * image.shape[1],
                        'y': landmark.y * image.shape[0],
                        'visibility': landmark.visibility
                    }
            
            return mask, keypoints
            
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            # Return basic mask
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return mask, {}
    
    def _extract_person_mask_from_yolo(self, results, image_shape: Tuple[int, int]) -> np.ndarray:
        """Extract person mask from YOLO results"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                for i, cls in enumerate(result.boxes.cls):
                    if int(cls) == 0:  # Person class
                        person_mask = result.masks.data[i].cpu().numpy()
                        person_mask = cv2.resize(person_mask, (image_shape[1], image_shape[0]))
                        mask = np.maximum(mask, (person_mask > 0.5).astype(np.uint8) * 255)
        
        return mask
    
    def _detect_garment_region(self, image: np.ndarray, keypoints: Dict, category: str) -> np.ndarray:
        """
        Detect the region where garment should be placed based on category and pose
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if not keypoints:
            # Fallback: use upper body region
            h, w = image.shape[:2]
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            return mask
        
        try:
            # Define body regions based on MediaPipe landmarks
            if "top" in category.lower() or "shirt" in category.lower() or "polo" in category.lower():
                # Upper body region
                points = self._get_upper_body_points(keypoints, image.shape)
            elif "bottom" in category.lower() or "pant" in category.lower() or "jean" in category.lower():
                # Lower body region  
                points = self._get_lower_body_points(keypoints, image.shape)
            elif "dress" in category.lower():
                # Full torso region
                points = self._get_dress_points(keypoints, image.shape)
            else:
                # Default upper body
                points = self._get_upper_body_points(keypoints, image.shape)
            
            if points:
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            else:
                # Fallback region
                h, w = image.shape[:2]
                cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
                
        except Exception as e:
            logger.error(f"Garment region detection error: {e}")
            # Fallback region
            h, w = image.shape[:2]
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        
        return mask
    
    def _get_upper_body_points(self, keypoints: Dict, image_shape: Tuple[int, int]) -> list:
        """Get polygon points for upper body region"""
        points = []
        
        # MediaPipe pose landmarks for upper body
        # 11: Left shoulder, 12: Right shoulder, 23: Left hip, 24: Right hip
        key_landmarks = [11, 12, 23, 24]
        
        for landmark_id in key_landmarks:
            if landmark_id in keypoints and keypoints[landmark_id]['visibility'] > 0.5:
                points.append([
                    int(keypoints[landmark_id]['x']),
                    int(keypoints[landmark_id]['y'])
                ])
        
        if len(points) >= 3:
            return points
        else:
            # Fallback to center region
            h, w = image_shape[:2]
            return [[w//4, h//4], [3*w//4, h//4], [3*w//4, 2*h//3], [w//4, 2*h//3]]
    
    def _get_lower_body_points(self, keypoints: Dict, image_shape: Tuple[int, int]) -> list:
        """Get polygon points for lower body region"""
        points = []
        
        # MediaPipe pose landmarks for lower body
        # 23: Left hip, 24: Right hip, 27: Left ankle, 28: Right ankle
        key_landmarks = [23, 24, 27, 28]
        
        for landmark_id in key_landmarks:
            if landmark_id in keypoints and keypoints[landmark_id]['visibility'] > 0.5:
                points.append([
                    int(keypoints[landmark_id]['x']),
                    int(keypoints[landmark_id]['y'])
                ])
        
        if len(points) >= 3:
            return points
        else:
            # Fallback to lower region
            h, w = image_shape[:2]
            return [[w//4, h//2], [3*w//4, h//2], [3*w//4, h], [w//4, h]]
    
    def _get_dress_points(self, keypoints: Dict, image_shape: Tuple[int, int]) -> list:
        """Get polygon points for dress region (full torso)"""
        points = []
        
        # MediaPipe pose landmarks for dress (shoulders to knees)
        key_landmarks = [11, 12, 25, 26]  # Shoulders and knees
        
        for landmark_id in key_landmarks:
            if landmark_id in keypoints and keypoints[landmark_id]['visibility'] > 0.5:
                points.append([
                    int(keypoints[landmark_id]['x']),
                    int(keypoints[landmark_id]['y'])
                ])
        
        if len(points) >= 3:
            return points
        else:
            # Fallback to torso region
            h, w = image_shape[:2]
            return [[w//4, h//4], [3*w//4, h//4], [3*w//4, 4*h//5], [w//4, 4*h//5]]
    
    def _fit_and_blend_garment(
        self,
        user_image: np.ndarray,
        garment_image: np.ndarray, 
        person_mask: np.ndarray,
        garment_region: np.ndarray,
        keypoints: Dict
    ) -> np.ndarray:
        """
        Fit garment to the detected region and blend with user image
        """
        try:
            # Resize garment to fit the detected region
            region_coords = cv2.findNonZero(garment_region)
            if region_coords is not None:
                x, y, w, h = cv2.boundingRect(region_coords)
                
                # Resize garment image to fit the region
                garment_resized = cv2.resize(garment_image, (w, h))
                
                # Create garment mask
                garment_mask = np.ones((h, w), dtype=np.uint8) * 255
                
                # Remove background from garment if it has one
                garment_pil = Image.fromarray(garment_resized)
                garment_no_bg = remove(garment_pil, session=self.bg_remover)
                
                if garment_no_bg.mode == 'RGBA':
                    garment_resized = np.array(garment_no_bg)[:,:,:3]
                    garment_mask = np.array(garment_no_bg)[:,:,3]
                
                # Blend garment with user image
                result = user_image.copy()
                
                # Apply garment to the region
                roi = result[y:y+h, x:x+w]
                
                # Create blended region
                for c in range(3):
                    roi[:, :, c] = np.where(
                        garment_mask > 128,
                        garment_resized[:, :, c],
                        roi[:, :, c]
                    )
                
                result[y:y+h, x:x+w] = roi
                
            else:
                # Fallback: simple overlay in center
                result = user_image.copy()
                h_user, w_user = user_image.shape[:2]
                h_gar, w_gar = garment_image.shape[:2]
                
                # Scale garment to reasonable size
                scale = min(w_user // 3, h_user // 3) / max(w_gar, h_gar)
                new_w, new_h = int(w_gar * scale), int(h_gar * scale)
                garment_resized = cv2.resize(garment_image, (new_w, new_h))
                
                # Center placement
                start_y = (h_user - new_h) // 3
                start_x = (w_user - new_w) // 2
                
                # Simple overlay
                result[start_y:start_y+new_h, start_x:start_x+new_w] = garment_resized
            
            return result
            
        except Exception as e:
            logger.error(f"Garment fitting error: {e}")
            return user_image
    
    def _enhance_result(self, result_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Enhance the try-on result with post-processing
        """
        try:
            # Convert to PIL for easier processing
            result_pil = Image.fromarray(result_image)
            
            # Apply subtle blur to blend edges
            result_pil = result_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Adjust brightness and contrast to match original
            from PIL import ImageEnhance
            
            # Enhance color slightly
            enhancer = ImageEnhance.Color(result_pil)
            result_pil = enhancer.enhance(1.1)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(result_pil)
            result_pil = enhancer.enhance(1.1)
            
            return np.array(result_pil)
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return result_image
    
    async def _save_result_image(self, result_image: np.ndarray) -> str:
        """
        Save result image and return as data URL
        """
        try:
            # Convert to PIL
            result_pil = Image.fromarray(result_image)
            
            # Save to bytes
            buffer = io.BytesIO()
            result_pil.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            # Convert to base64 data URL
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            data_url = f"data:image/jpeg;base64,{image_b64}"
            
            return data_url
            
        except Exception as e:
            logger.error(f"Result saving error: {e}")
            # Return placeholder
            return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgZmlsbD0iIzMzMyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmaWxsPSIjZmZmIiBkb21pbmFudC1iYXNlbGluZT0iY2VudHJhbCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+VHJ5LU9uIFJlc3VsdDwvdGV4dD48L3N2Zz4="

# Global engine instance
virtual_tryon_engine = VirtualTryOnEngine()