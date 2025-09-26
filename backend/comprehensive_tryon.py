import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import logging
from dataclasses import dataclass

class GarmentType(Enum):
    """Supported garment types for virtual try-on"""
    TOP = "top"                    # T-shirts, shirts, blouses, sweaters
    BOTTOM = "bottom"              # Pants, jeans, shorts, skirts
    SHOES = "shoes"                # All footwear
    DRESS = "dress"                # Full dresses
    OUTERWEAR = "outerwear"        # Jackets, coats
    COMBINATION = "combination"     # Multiple items together

@dataclass
class ProcessingResult:
    """Result of virtual try-on processing"""
    success: bool
    result_image: Optional[np.ndarray] = None
    modified_regions: List[str] = None
    preserved_regions: List[str] = None
    quality_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class ComprehensiveRegionTryOn:
    """
    Production-ready virtual try-on system that handles multiple garment types
    by modifying only specific body regions while preserving everything else
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Region definitions for different garment types
        self.region_definitions = {
            GarmentType.TOP: {
                'primary_landmarks': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
                'secondary_landmarks': ['left_elbow', 'right_elbow'],  # For sleeve boundaries
                'exclude_regions': ['arms', 'face', 'legs'],
                'mask_type': 'torso',
                'preserve_neckline': True
            },
            GarmentType.BOTTOM: {
                'primary_landmarks': ['left_hip', 'right_hip', 'left_ankle', 'right_ankle'],
                'secondary_landmarks': ['left_knee', 'right_knee'],
                'exclude_regions': ['torso', 'face', 'arms'],
                'mask_type': 'lower_body',
                'preserve_crotch': True
            },
            GarmentType.SHOES: {
                'primary_landmarks': ['left_ankle', 'right_ankle'],
                'secondary_landmarks': [],
                'exclude_regions': ['legs', 'torso', 'face', 'arms'],
                'mask_type': 'feet',
                'preserve_ankle': True
            },
            GarmentType.DRESS: {
                'primary_landmarks': ['left_shoulder', 'right_shoulder', 'left_ankle', 'right_ankle'],
                'secondary_landmarks': ['left_hip', 'right_hip'],
                'exclude_regions': ['arms', 'face'],
                'mask_type': 'full_dress',
                'preserve_neckline': True
            },
            GarmentType.OUTERWEAR: {
                'primary_landmarks': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
                'secondary_landmarks': ['left_wrist', 'right_wrist'],  # Include sleeves
                'exclude_regions': ['face', 'legs'],
                'mask_type': 'torso_with_arms',
                'preserve_neckline': True
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'landmark_confidence': 0.7,
            'minimum_mask_size': 1000,  # pixels
            'maximum_mask_size': 100000,  # pixels
            'blend_quality': 0.8
        }
        
    def process_virtual_tryon(self, customer_analysis: Dict, garment_analysis: Dict, 
                            product_info: Dict, original_image: np.ndarray,
                            garment_types: List[str]) -> ProcessingResult:
        """
        Main processing function for comprehensive virtual try-on
        
        Args:
            customer_analysis: Body measurements and pose data from your existing system
            garment_analysis: Garment properties from your existing system  
            product_info: Product metadata from your existing system
            original_image: Customer photo as numpy array
            garment_types: List of garment types to process (e.g., ['top'], ['bottom'], ['top', 'bottom'])
        """
        start_time = cv2.getTickCount()
        
        try:
            self.logger.info(f"[COMPREHENSIVE] Starting virtual try-on for {garment_types}")
            
            # Validate inputs
            validation_result = self._validate_inputs(customer_analysis, original_image, garment_types)
            if not validation_result['valid']:
                return ProcessingResult(
                    success=False,
                    error_message=validation_result['error']
                )
            
            # Convert garment types to enums
            garment_enums = [self._get_garment_type(gt) for gt in garment_types]
            
            # Create processing plan
            processing_plan = self._create_processing_plan(garment_enums, customer_analysis)
            self.logger.info(f"[PLAN] Processing plan: {len(processing_plan)} regions")
            
            # Start with original image
            result_image = original_image.copy()
            all_modified_regions = []
            all_preserved_regions = ['background']  # Always preserve background
            
            # Process each garment type
            for garment_type in garment_enums:
                self.logger.info(f"[PROCESSING] Processing {garment_type.value}")
                
                # Create region mask
                region_mask = self._create_region_mask(
                    garment_type, customer_analysis, original_image.shape
                )
                
                if region_mask is None:
                    self.logger.warning(f"[SKIP] Could not create mask for {garment_type.value}")
                    continue
                
                # Generate garment appearance
                garment_region = self._generate_garment_appearance(
                    garment_type, garment_analysis, product_info, original_image.shape, region_mask
                )
                
                # Apply realistic effects
                enhanced_garment = self._apply_realistic_effects(
                    garment_region, original_image, region_mask, garment_type, customer_analysis
                )
                
                # Blend with existing result
                result_image = self._blend_region_with_image(
                    result_image, enhanced_garment, region_mask, garment_type
                )
                
                # Track regions
                all_modified_regions.append(garment_type.value)
                all_preserved_regions.extend(self.region_definitions[garment_type]['exclude_regions'])
            
            # Remove duplicates and add standard preserved regions
            all_preserved_regions = list(set(all_preserved_regions + ['face', 'background']))
            if 'top' not in garment_types:
                all_preserved_regions.append('torso')
            if 'bottom' not in garment_types:
                all_preserved_regions.append('legs')
            if 'shoes' not in garment_types:
                all_preserved_regions.append('feet')
            
            # Final quality enhancement
            final_result = self._final_quality_enhancement(result_image, original_image)
            
            # Calculate metrics
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            quality_score = self._calculate_quality_score(final_result, original_image, all_modified_regions)
            
            self.logger.info(f"[SUCCESS] Virtual try-on completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                result_image=final_result,
                modified_regions=all_modified_regions,
                preserved_regions=all_preserved_regions,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata={
                    'garment_types_processed': garment_types,
                    'total_regions_modified': len(all_modified_regions),
                    'landmark_confidence': np.mean([
                        customer_analysis.get('pose_landmarks', {}).get(lm, {}).get('confidence', 0)
                        for lm in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
                    ])
                }
            )
            
        except Exception as e:
            self.logger.error(f"[ERROR] Virtual try-on failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=f"Processing failed: {str(e)}"
            )
    
    def _validate_inputs(self, customer_analysis: Dict, original_image: np.ndarray, 
                        garment_types: List[str]) -> Dict[str, Any]:
        """Validate all inputs before processing"""
        
        # Check image
        if original_image is None or original_image.size == 0:
            return {'valid': False, 'error': 'Invalid original image'}
        
        # Check required landmarks exist
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        missing_landmarks = []
        for landmark in required_landmarks:
            if landmark not in pose_landmarks or pose_landmarks[landmark].get('confidence', 0) < self.quality_thresholds['landmark_confidence']:
                missing_landmarks.append(landmark)
        
        if missing_landmarks:
            return {'valid': False, 'error': f'Missing or low confidence landmarks: {missing_landmarks}'}
        
        # Check garment types
        valid_types = [gt.value for gt in GarmentType if gt != GarmentType.COMBINATION]
        invalid_types = [gt for gt in garment_types if gt not in valid_types]
        
        if invalid_types:
            return {'valid': False, 'error': f'Invalid garment types: {invalid_types}'}
        
        return {'valid': True}
    
    def _get_garment_type(self, garment_string: str) -> GarmentType:
        """Convert string to GarmentType enum"""
        type_mapping = {
            'top': GarmentType.TOP,
            'shirt': GarmentType.TOP,
            'tshirt': GarmentType.TOP,
            't-shirt': GarmentType.TOP,
            'blouse': GarmentType.TOP,
            'sweater': GarmentType.TOP,
            'bottom': GarmentType.BOTTOM,
            'pants': GarmentType.BOTTOM,
            'jeans': GarmentType.BOTTOM,
            'shorts': GarmentType.BOTTOM,
            'skirt': GarmentType.BOTTOM,
            'shoes': GarmentType.SHOES,
            'sneakers': GarmentType.SHOES,
            'boots': GarmentType.SHOES,
            'dress': GarmentType.DRESS,
            'outerwear': GarmentType.OUTERWEAR,
            'jacket': GarmentType.OUTERWEAR,
            'coat': GarmentType.OUTERWEAR
        }
        return type_mapping.get(garment_string.lower(), GarmentType.TOP)
    
    def _create_processing_plan(self, garment_types: List[GarmentType], 
                              customer_analysis: Dict) -> List[Dict]:
        """Create processing plan for multiple garments"""
        plan = []
        
        # Define processing order (important for layering)
        processing_order = [GarmentType.DRESS, GarmentType.BOTTOM, GarmentType.TOP, GarmentType.SHOES, GarmentType.OUTERWEAR]
        
        for garment_type in processing_order:
            if garment_type in garment_types:
                plan.append({
                    'garment_type': garment_type,
                    'region_def': self.region_definitions[garment_type],
                    'priority': processing_order.index(garment_type)
                })
        
        return sorted(plan, key=lambda x: x['priority'])
    
    def _create_region_mask(self, garment_type: GarmentType, customer_analysis: Dict, 
                           image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Create precise mask for specific garment region"""
        
        height, width, _ = image_shape
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        region_def = self.region_definitions[garment_type]
        
        self.logger.info(f"[MASK] Creating {region_def['mask_type']} mask for {garment_type.value}")
        
        if garment_type == GarmentType.TOP:
            return self._create_top_mask(pose_landmarks, (height, width))
        elif garment_type == GarmentType.BOTTOM:
            return self._create_bottom_mask(pose_landmarks, (height, width))
        elif garment_type == GarmentType.SHOES:
            return self._create_shoes_mask(pose_landmarks, (height, width))
        elif garment_type == GarmentType.DRESS:
            return self._create_dress_mask(pose_landmarks, (height, width))
        elif garment_type == GarmentType.OUTERWEAR:
            return self._create_outerwear_mask(pose_landmarks, (height, width))
        
        return None
    
    def _create_top_mask(self, pose_landmarks: Dict, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create enhanced mask for top garments - FIXED VERSION"""
        
        height, width = image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get required landmarks
        required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        points = {}
        
        for landmark in required_landmarks:
            if landmark in pose_landmarks:
                lm_data = pose_landmarks[landmark]
                if isinstance(lm_data, (list, tuple)) and len(lm_data) >= 2:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(lm_data[0] * width)
                    y = int(lm_data[1] * height)
                    points[landmark] = (x, y)
                elif isinstance(lm_data, dict) and 'x' in lm_data and 'y' in lm_data:
                    x = int(lm_data['x'] * width)
                    y = int(lm_data['y'] * height)
                    points[landmark] = (x, y)
        
        if len(points) < 4:
            self.logger.error(f"[MASK] Insufficient landmarks for top mask: {list(points.keys())}")
            return None
        
        # Calculate torso region with expanded boundaries
        left_shoulder = points['left_shoulder']
        right_shoulder = points['right_shoulder']
        left_hip = points['left_hip']
        right_hip = points['right_hip']
        
        # Expand the region to ensure full coverage
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        torso_expansion = max(20, int(shoulder_width * 0.15))  # Minimum 20px expansion
        
        # Create expanded polygon points
        polygon_points = [
            # Top shoulder line (expanded outward)
            (left_shoulder[0] - torso_expansion, left_shoulder[1] - int(torso_expansion * 0.5)),
            (right_shoulder[0] + torso_expansion, right_shoulder[1] - int(torso_expansion * 0.5)),
            
            # Side expansion at mid-torso
            (right_shoulder[0] + torso_expansion, int((right_shoulder[1] + right_hip[1]) / 2)),
            
            # Bottom hip line (expanded outward)
            (right_hip[0] + torso_expansion, right_hip[1] + int(torso_expansion * 0.3)),
            (left_hip[0] - torso_expansion, left_hip[1] + int(torso_expansion * 0.3)),
            
            # Side expansion at mid-torso (left side)
            (left_shoulder[0] - torso_expansion, int((left_shoulder[1] + left_hip[1]) / 2)),
        ]
        
        # Convert to numpy array and ensure within image bounds
        polygon_points = np.array(polygon_points, dtype=np.int32)
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, width - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, height - 1)
        
        # Fill the polygon
        cv2.fillPoly(mask, [polygon_points], 255)
        
        # Apply smoothing to create more natural edges
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        
        # Ensure minimum mask size
        mask_area = np.sum(mask > 128)
        min_area = (width * height) * 0.05  # At least 5% of image
        
        if mask_area < min_area:
            self.logger.warning(f"[MASK] Top mask too small ({mask_area} < {min_area}), expanding...")
            # Dilate to expand
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Log mask statistics for debugging
        final_area = np.sum(mask > 128)
        coverage_percent = (final_area / (width * height)) * 100
        
        self.logger.info(f"[MASK] Top mask created: {final_area} pixels ({coverage_percent:.1f}% coverage)")
        self.logger.info(f"[MASK] Landmarks used: {list(points.keys())}")
        self.logger.info(f"[MASK] Expansion applied: {torso_expansion} pixels")
        
        # Debug: Save mask for inspection (remove in production)
        cv2.imwrite('/tmp/debug_top_mask.png', mask)
        
        return mask
    
    def _create_bottom_mask(self, pose_landmarks: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create mask for bottoms (pants, jeans, shorts, skirts)"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get landmarks
        lh = pose_landmarks.get('left_hip', {})
        rh = pose_landmarks.get('right_hip', {})
        lk = pose_landmarks.get('left_knee', {})
        rk = pose_landmarks.get('right_knee', {})
        la = pose_landmarks.get('left_ankle', {})
        ra = pose_landmarks.get('right_ankle', {})
        
        # Minimum requirement: hips and knees
        if all(point.get('confidence', 0) > 0.5 for point in [lh, rh, lk, rk]):
            # Convert to pixels
            lh_x, lh_y = int(lh['x'] * width), int(lh['y'] * height)
            rh_x, rh_y = int(rh['x'] * width), int(rh['y'] * height)
            lk_x, lk_y = int(lk['x'] * width), int(lk['y'] * height)
            rk_x, rk_y = int(rk['x'] * width), int(rk['y'] * height)
            
            # Use ankles if available, otherwise estimate
            if la.get('confidence', 0) > 0.3 and ra.get('confidence', 0) > 0.3:
                la_x, la_y = int(la['x'] * width), int(la['y'] * height)
                ra_x, ra_y = int(ra['x'] * width), int(ra['y'] * height)
            else:
                # Estimate ankle positions
                leg_length = abs(lk_y - lh_y) * 1.2
                la_x, la_y = lk_x, int(lk_y + leg_length)
                ra_x, ra_y = rk_x, int(rk_y + leg_length)
            
            # Create lower body polygon
            hip_width = abs(rh_x - lh_x)
            ankle_width = int(hip_width * 0.4)  # Narrower at ankles
            
            points = np.array([
                [lh_x, lh_y],                           # Left hip
                [rh_x, rh_y],                           # Right hip
                [ra_x + ankle_width//2, ra_y],          # Right ankle outer
                [ra_x - ankle_width//2, ra_y],          # Right ankle inner
                [la_x + ankle_width//2, la_y],          # Left ankle inner
                [la_x - ankle_width//2, la_y],          # Left ankle outer
            ], dtype=np.int32)
            
            cv2.fillPoly(mask, [points], 255)
            
            # Smooth and refine
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (9, 9), 3)
            
            self.logger.info(f"[BOTTOM-MASK] Created lower body mask: {np.sum(mask > 0)} pixels")
            return mask
            
        return None
    
    def _create_shoes_mask(self, pose_landmarks: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create mask for shoes"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get ankle landmarks
        la = pose_landmarks.get('left_ankle', {})
        ra = pose_landmarks.get('right_ankle', {})
        
        if all(point.get('confidence', 0) > 0.5 for point in [la, ra]):
            # Convert to pixels
            la_x, la_y = int(la['x'] * width), int(la['y'] * height)
            ra_x, ra_y = int(ra['x'] * width), int(ra['y'] * height)
            
            # Define shoe dimensions (approximate)
            shoe_width = int(width * 0.08)   # 8% of image width
            shoe_height = int(height * 0.06) # 6% of image height
            
            # Create foot regions
            for ankle_x, ankle_y in [(la_x, la_y), (ra_x, ra_y)]:
                # Shoe extends down from ankle and slightly forward
                foot_top = ankle_y - shoe_height // 4
                foot_bottom = ankle_y + int(shoe_height * 0.75)
                foot_left = ankle_x - shoe_width // 2
                foot_right = ankle_x + shoe_width // 2
                
                # Create shoe rectangle
                cv2.rectangle(mask, (foot_left, foot_top), (foot_right, foot_bottom), 255, -1)
            
            # Smooth shoe regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (7, 7), 2)
            
            self.logger.info(f"[SHOES-MASK] Created shoes mask: {np.sum(mask > 0)} pixels")
            return mask
            
        return None
    
    def _create_dress_mask(self, pose_landmarks: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create mask for dresses (combines torso and lower body)"""
        height, width = image_shape
        
        # Combine top and bottom masks for dress
        top_mask = self._create_top_mask(pose_landmarks, image_shape)
        bottom_mask = self._create_bottom_mask(pose_landmarks, image_shape)
        
        if top_mask is not None and bottom_mask is not None:
            # Combine masks
            dress_mask = cv2.bitwise_or(top_mask, bottom_mask)
            
            # Smooth the combined mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            dress_mask = cv2.morphologyEx(dress_mask, cv2.MORPH_CLOSE, kernel)
            dress_mask = cv2.GaussianBlur(dress_mask, (15, 15), 5)
            
            self.logger.info(f"[DRESS-MASK] Created dress mask: {np.sum(dress_mask > 0)} pixels")
            return dress_mask
            
        return None
    
    def _create_outerwear_mask(self, pose_landmarks: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create mask for outerwear (includes arms/sleeves)"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get all relevant landmarks
        ls = pose_landmarks.get('left_shoulder', {})
        rs = pose_landmarks.get('right_shoulder', {})
        le = pose_landmarks.get('left_elbow', {})
        re = pose_landmarks.get('right_elbow', {})
        lw = pose_landmarks.get('left_wrist', {})
        rw = pose_landmarks.get('right_wrist', {})
        lh = pose_landmarks.get('left_hip', {})
        rh = pose_landmarks.get('right_hip', {})
        
        if all(point.get('confidence', 0) > 0.6 for point in [ls, rs, lh, rh]):
            # Start with torso mask
            torso_mask = self._create_top_mask(pose_landmarks, image_shape)
            if torso_mask is not None:
                mask = torso_mask.copy()
            
            # Add sleeves if arm landmarks are available
            if all(point.get('confidence', 0) > 0.5 for point in [le, re]):
                # Convert to pixels
                ls_x, ls_y = int(ls['x'] * width), int(ls['y'] * height)
                rs_x, rs_y = int(rs['x'] * width), int(rs['y'] * height)
                le_x, le_y = int(le['x'] * width), int(le['y'] * height)
                re_x, re_y = int(re['x'] * width), int(re['y'] * height)
                
                # Use wrists if available
                if lw.get('confidence', 0) > 0.4:
                    lw_x, lw_y = int(lw['x'] * width), int(lw['y'] * height)
                else:
                    lw_x, lw_y = le_x, le_y + 40  # Estimate wrist position
                    
                if rw.get('confidence', 0) > 0.4:
                    rw_x, rw_y = int(rw['x'] * width), int(rw['y'] * height)
                else:
                    rw_x, rw_y = re_x, re_y + 40  # Estimate wrist position
                
                # Create sleeve polygons
                sleeve_width = 25  # Sleeve thickness
                
                # Left sleeve
                left_sleeve = np.array([
                    [ls_x - sleeve_width, ls_y],
                    [ls_x + sleeve_width, ls_y],
                    [le_x + sleeve_width, le_y],
                    [lw_x + sleeve_width, lw_y],
                    [lw_x - sleeve_width, lw_y],
                    [le_x - sleeve_width, le_y]
                ], dtype=np.int32)
                
                # Right sleeve
                right_sleeve = np.array([
                    [rs_x - sleeve_width, rs_y],
                    [rs_x + sleeve_width, rs_y],
                    [re_x + sleeve_width, re_y],
                    [rw_x + sleeve_width, rw_y],
                    [rw_x - sleeve_width, rw_y],
                    [re_x - sleeve_width, re_y]
                ], dtype=np.int32)
                
                # Add sleeves to mask
                cv2.fillPoly(mask, [left_sleeve], 255)
                cv2.fillPoly(mask, [right_sleeve], 255)
            
            # Final smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (13, 13), 4)
            
            self.logger.info(f"[OUTERWEAR-MASK] Created outerwear mask: {np.sum(mask > 0)} pixels")
            return mask
            
        return None
    
    def _generate_garment_appearance(self, garment_type: GarmentType, garment_analysis: Dict,
                                   product_info: Dict, image_shape: Tuple[int, int, int],
                                   mask: np.ndarray) -> np.ndarray:
        """Generate realistic garment appearance"""
        
        height, width, channels = image_shape
        garment_region = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Get garment properties
        dominant_colors = garment_analysis.get('dominant_colors', [(128, 128, 128)])
        texture_features = garment_analysis.get('texture_features', {})
        
        # Extract data
        dominant_colors = garment_analysis.get('dominant_colors', [])
        texture_features = garment_analysis.get('texture_features', {})
        product_name = product_info.get('name', '').lower()
        
        # FIXED: Priority order - product name first, then analysis
        base_color = None
        
        # 1. First check product name (highest priority)
        if 'white' in product_name or 'blanc' in product_name:
            base_color = (255, 255, 255)
            self.logger.info(f"[GARMENT] Using WHITE from product name: {product_name}")
        elif 'black' in product_name or 'noir' in product_name:
            base_color = (0, 0, 0)
            self.logger.info(f"[GARMENT] Using BLACK from product name: {product_name}")
        elif 'blue' in product_name or 'bleu' in product_name:
            base_color = (100, 50, 200)
            self.logger.info(f"[GARMENT] Using BLUE from product name: {product_name}")
        elif 'red' in product_name or 'rouge' in product_name:
            base_color = (200, 50, 50)
            self.logger.info(f"[GARMENT] Using RED from product name: {product_name}")
        
        # 2. If no color in name, check for actual white pixels in analysis
        if base_color is None and dominant_colors:
            # Check if any dominant color is actually white-ish
            for color in dominant_colors:
                r, g, b = color
                # Check if color is close to white (all values > 200)
                if r > 200 and g > 200 and b > 200:
                    base_color = (255, 255, 255)  # Force pure white
                    self.logger.info(f"[GARMENT] Detected white-ish color {color}, using pure WHITE")
                    break
        
        # 3. Fallback to dominant color analysis
        if base_color is None:
            base_color = dominant_colors[0] if dominant_colors else (128, 128, 128)
            self.logger.info(f"[GARMENT] Using analyzed dominant color: {base_color}")
        
        # Fill base color
        garment_region[:] = base_color
        
        # Add fabric texture (reduced intensity for white garments)
        roughness = texture_features.get('roughness', 0.3)
        complexity = texture_features.get('complexity', 0.1)
        
        # Reduce texture for white/light colors to maintain clarity
        if base_color[0] > 200 and base_color[1] > 200 and base_color[2] > 200:
            roughness *= 0.3  # Much less texture for white garments
            self.logger.info(f"[GARMENT] Reduced texture for light garment")
        
        # Add fabric texture
        if roughness > 0.1:
            noise_strength = max(1, int(roughness * 10))  # Minimum noise
            noise = np.random.normal(0, noise_strength, garment_region.shape)
            garment_region = np.clip(
                garment_region.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
        
        # Add garment-specific details
        if garment_type == GarmentType.SHOES:
            garment_region = self._add_shoe_details(garment_region, mask)
        elif garment_type == GarmentType.BOTTOM:
            garment_region = self._add_pants_details(garment_region, mask)
        elif garment_type == GarmentType.TOP:
            garment_region = self._add_shirt_details(garment_region, mask, base_color)
        
        self.logger.info(f"[APPEARANCE] Generated {garment_type.value} with final color {base_color}")
        return garment_region
    
    def _add_shirt_details(self, garment_region: np.ndarray, mask: np.ndarray, base_color: Tuple) -> np.ndarray:
        """Add shirt-specific details like seams, collar hints"""
        
        # Add subtle seam lines for shirts
        height, width = garment_region.shape[:2]
        
        # Create slightly darker color for seams
        seam_color = tuple(max(0, int(c * 0.95)) for c in base_color)
        
        # Add vertical seam lines at sides
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum torso size
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add side seams
                left_seam_x = x + int(w * 0.1)
                right_seam_x = x + int(w * 0.9)
                
                # Draw subtle seam lines
                cv2.line(garment_region, (left_seam_x, y), (left_seam_x, y + h), seam_color, 1)
                cv2.line(garment_region, (right_seam_x, y), (right_seam_x, y + h), seam_color, 1)
        
        return garment_region
    
    def _add_garment_patterns(self, garment_region: np.ndarray, secondary_colors: List[Tuple], 
                            complexity: float) -> None:
        """Add patterns or secondary colors to garment"""
        if complexity > 0.1 and secondary_colors:
            # Add subtle color variations
            height, width = garment_region.shape[:2]
            pattern_intensity = min(complexity * 0.3, 0.1)  # Max 10% blend
            
            # Create simple pattern overlay
            for i, color in enumerate(secondary_colors[:2]):  # Max 2 secondary colors
                pattern = np.random.rand(height, width) < (pattern_intensity * (1 - i * 0.5))
                garment_region[pattern] = color
    
    def _add_shoe_details(self, garment_region: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add shoe-specific details like laces, soles, etc."""
        # Add darker sole area
        sole_color = tuple(int(c * 0.7) for c in garment_region[0, 0])
        
        # Find bottom 30% of each shoe region for sole
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum shoe size
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                sole_height = int(h * 0.3)
                sole_region = garment_region[y + h - sole_height:y + h, x:x + w]
                sole_region[:] = sole_color
        
        return garment_region
    
    def _add_pants_details(self, garment_region: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add pants-specific details like seams"""
        # Add subtle seam lines (darker lines along legs)
        height, width = garment_region.shape[:2]
        
        # Create vertical seam lines
        seam_color = tuple(int(c * 0.9) for c in garment_region[0, 0])
        
        # Find center lines of legs for seams
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum leg size
                # Get center line of contour
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    # Draw subtle seam line
                    cv2.line(garment_region, (cx, 0), (cx, height), seam_color, 1)
        
        return garment_region
    
    def _apply_realistic_effects(self, garment_region: np.ndarray, original_image: np.ndarray,
                               mask: np.ndarray, garment_type: GarmentType,
                               customer_analysis: Dict) -> np.ndarray:
        """Apply realistic lighting, shadows, and body curvature effects"""
        
        # Apply original image lighting
        enhanced_garment = self._apply_lighting(garment_region, original_image, mask)
        
        # Add body curvature effects
        enhanced_garment = self._apply_body_curvature(
            enhanced_garment, mask, garment_type, customer_analysis
        )
        
        # Add shadows and depth
        enhanced_garment = self._add_shadows_and_depth(enhanced_garment, mask, garment_type)
        
        return enhanced_garment
    
    def _apply_lighting(self, garment_region: np.ndarray, original_image: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
        """Apply realistic lighting from original image"""
        
        # Extract lighting from original image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        masked_lighting = cv2.bitwise_and(original_gray, original_gray, mask=mask)
        
        # Create smooth lighting map
        lighting_map = cv2.GaussianBlur(masked_lighting, (25, 25), 8)
        lighting_map = lighting_map.astype(np.float32) / 255.0
        
        # Apply lighting to garment
        enhanced = garment_region.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] = enhanced[:, :, c] * (lighting_map * 1.4 + 0.3)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_body_curvature(self, garment_region: np.ndarray, mask: np.ndarray,
                            garment_type: GarmentType, customer_analysis: Dict) -> np.ndarray:
        """Apply body curvature effects for realism"""
        
        # Get body measurements for curvature calculation
        measurements = customer_analysis.get('measurements', {})
        
        if garment_type == GarmentType.TOP:
            # Apply chest curvature
            chest_cm = measurements.get('chest_circumference_cm', 90)
            curvature_factor = max(0.8, min(1.2, chest_cm / 90.0))  # Normalize around 90cm
        else:
            curvature_factor = 1.0
        
        # Apply subtle barrel distortion for body curvature
        if curvature_factor != 1.0:
            height, width = garment_region.shape[:2]
            distortion_strength = (curvature_factor - 1.0) * 0.1  # Subtle effect
            
            # Create distortion map
            center_x, center_y = width // 2, height // 2
            y, x = np.mgrid[0:height, 0:width]
            
            # Apply radial distortion
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            r_max = min(width, height) // 2
            r_norm = r / r_max
            
            distortion = 1 + distortion_strength * r_norm
            
            # Apply distortion only in masked regions
            distorted = garment_region.copy()
            mask_bool = mask > 128
            
            if np.any(mask_bool):
                for c in range(3):
                    channel = garment_region[:, :, c].astype(np.float32)
                    distorted[:, :, c] = np.where(
                        mask_bool,
                        np.clip(channel * distortion, 0, 255),
                        channel
                    ).astype(np.uint8)
            
            return distorted
        
        return garment_region
    
    def _add_shadows_and_depth(self, garment_region: np.ndarray, mask: np.ndarray,
                             garment_type: GarmentType) -> np.ndarray:
        """Add realistic shadows and depth effects"""
        
        # Create depth map based on distance from edges
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        distance_transform = cv2.GaussianBlur(distance_transform, (15, 15), 5)
        
        # Normalize distance transform
        if np.max(distance_transform) > 0:
            distance_transform = distance_transform / np.max(distance_transform)
        
        # Apply depth-based shading
        depth_shading = 0.85 + 0.15 * distance_transform  # Darker at edges, lighter in center
        
        enhanced = garment_region.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] = enhanced[:, :, c] * depth_shading
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _blend_region_with_image(self, base_image: np.ndarray, garment_region: np.ndarray,
                               mask: np.ndarray, garment_type: GarmentType) -> np.ndarray:
        """Enhanced blending with better edge handling - FIXED VERSION"""
        
        # Ensure mask is proper format
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Create smooth transition mask with stronger blending
        smooth_mask = cv2.GaussianBlur(mask, (31, 31), 15).astype(np.float32) / 255.0
        
        # Ensure we have a significant blending region
        blend_threshold = 0.1  # Minimum blend strength
        smooth_mask = np.maximum(smooth_mask, blend_threshold * (mask > 128).astype(np.float32))
        
        # Convert images to float for blending
        result = base_image.copy().astype(np.float32)
        garment_float = garment_region.astype(np.float32)
        
        # Perform alpha blending
        for c in range(3):
            result[:, :, c] = (
                base_image[:, :, c].astype(np.float32) * (1 - smooth_mask) +
                garment_float[:, :, c] * smooth_mask
            )
        
        # Convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Log blending statistics
        blend_area = np.sum(smooth_mask > 0.1)
        total_area = mask.shape[0] * mask.shape[1]
        blend_percent = (blend_area / total_area) * 100
        
        self.logger.info(f"[BLEND] Blended {blend_area} pixels ({blend_percent:.1f}% of image)")
        self.logger.info(f"[BLEND] Max blend strength: {np.max(smooth_mask):.2f}")
        
        # Debug: Save blend mask for inspection (remove in production)
        cv2.imwrite('/tmp/debug_blend_mask.png', (smooth_mask * 255).astype(np.uint8))
        
        return result
    
    def _enhance_blend_edges(self, blended: np.ndarray, original: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """Enhance edges for seamless blending"""
        
        # Find mask edges
        edges = cv2.Canny((mask > 128).astype(np.uint8), 50, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        
        # Apply gentle blur to edges
        result = blended.copy()
        for c in range(3):
            edge_region = cv2.bitwise_and(result[:, :, c], result[:, :, c], mask=edges)
            blurred_edges = cv2.GaussianBlur(edge_region, (5, 5), 2)
            result[:, :, c] = np.where(edges > 0, blurred_edges, result[:, :, c])
        
        return result
    
    def _final_quality_enhancement(self, result_image: np.ndarray,
                                 original_image: np.ndarray) -> np.ndarray:
        """Apply final quality enhancements"""
        
        # Color temperature matching
        enhanced = self._match_color_temperature(result_image, original_image)
        
        # Noise matching
        enhanced = self._match_noise_characteristics(enhanced, original_image)
        
        # Final sharpening
        enhanced = self._apply_selective_sharpening(enhanced, original_image)
        
        return enhanced
    
    def _match_color_temperature(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Match color temperature between result and original"""
        # Simple color temperature matching
        result_mean = np.mean(result, axis=(0, 1))
        original_mean = np.mean(original, axis=(0, 1))
        
        if np.all(original_mean > 0):
            color_ratio = original_mean / result_mean
            # Apply gentle color correction
            color_ratio = np.clip(color_ratio, 0.9, 1.1)  # Limit correction strength
            
            corrected = result.astype(np.float32)
            for c in range(3):
                corrected[:, :, c] *= color_ratio[c]
            
            return np.clip(corrected, 0, 255).astype(np.uint8)
        
        return result
    
    def _match_noise_characteristics(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Match noise characteristics between result and original"""
        # Estimate noise level in original
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        noise_estimate = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        
        if noise_estimate > 100:  # If original has significant noise
            # Add subtle matching noise to result
            noise_strength = min(3, int(np.sqrt(noise_estimate) / 10))
            noise = np.random.normal(0, noise_strength, result.shape)
            result_with_noise = result.astype(np.float32) + noise
            return np.clip(result_with_noise, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_selective_sharpening(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply selective sharpening to match original image sharpness"""
        # Create unsharp mask
        blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
        sharpened = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        
        # Apply gentle sharpening
        return cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
    
    def _calculate_quality_score(self, result: np.ndarray, original: np.ndarray,
                               modified_regions: List[str]) -> float:
        """Enhanced quality score that actually measures visual transformation - FIXED VERSION"""
        
        scores = []
        
        # 1. Visual Change Score (CRITICAL - was missing before)
        # Measure actual pixel differences to ensure transformation occurred
        diff = cv2.absdiff(result, original)
        total_change = np.sum(diff)
        total_pixels = result.shape[0] * result.shape[1] * result.shape[2]
        
        # Normalize change score (expect at least some change for good quality)
        change_ratio = total_change / (total_pixels * 255.0)  # 0 to 1 scale
        
        if change_ratio < 0.01:  # Less than 1% change
            visual_change_score = 0.0  # No meaningful change detected
            self.logger.warning(f"[QUALITY] Very low visual change detected: {change_ratio:.4f}")
        elif change_ratio < 0.05:  # 1-5% change
            visual_change_score = 0.3  # Some change but minimal
        elif change_ratio < 0.15:  # 5-15% change
            visual_change_score = 0.7  # Good amount of change
        else:  # >15% change
            visual_change_score = 1.0  # Significant change
        
        scores.append(visual_change_score)
        self.logger.info(f"[QUALITY] Visual change score: {visual_change_score:.2f} (change ratio: {change_ratio:.4f})")
        
        # 2. Image Quality Score
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(result_gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 1000.0)
        scores.append(sharpness_score)
        self.logger.info(f"[QUALITY] Sharpness score: {sharpness_score:.2f}")
        
        # 3. Color Consistency Score
        result_std = np.std(result)
        original_std = np.std(original)
        if original_std > 0:
            color_consistency = 1.0 - min(1.0, abs(result_std - original_std) / original_std)
            scores.append(max(0.5, color_consistency))
            self.logger.info(f"[QUALITY] Color consistency score: {color_consistency:.2f}")
        
        # 4. Region Modification Appropriateness
        if len(modified_regions) == 0:
            modification_score = 0.0  # No regions modified = failure
        elif len(modified_regions) <= 2:
            modification_score = 1.0  # Appropriate number of regions
        else:
            modification_score = max(0.5, 1.0 - (len(modified_regions) - 2) * 0.2)
        
        scores.append(modification_score)
        self.logger.info(f"[QUALITY] Modification score: {modification_score:.2f} ({len(modified_regions)} regions)")
        
        # 5. Blending Quality (check for artifacts)
        # Look for harsh transitions that indicate poor blending
        edges = cv2.Canny(result_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (result.shape[0] * result.shape[1])
        
        if edge_density > 0.1:  # Too many edges might indicate artifacts
            blending_score = 0.6
        else:
            blending_score = 1.0
        
        scores.append(blending_score)
        self.logger.info(f"[QUALITY] Blending score: {blending_score:.2f} (edge density: {edge_density:.4f})")
        
        # Weight the scores (visual change is most important)
        weights = [0.4, 0.2, 0.15, 0.15, 0.1]  # Visual change gets 40% weight
        final_score = sum(score * weight for score, weight in zip(scores, weights))
        
        self.logger.info(f"[QUALITY] Component scores: {[f'{s:.2f}' for s in scores]}")
        self.logger.info(f"[QUALITY] Final weighted score: {final_score:.2f}")
        
        # Additional diagnostic info
        if final_score < 0.5:
            self.logger.warning("[QUALITY] Low quality score indicates processing issues")
            if visual_change_score < 0.3:
                self.logger.warning("[QUALITY] Primary issue: Insufficient visual transformation")
            if sharpness_score < 0.3:
                self.logger.warning("[QUALITY] Primary issue: Poor image sharpness")
        
        return float(final_score)
    
    def _debug_processing_pipeline(self, customer_analysis: Dict, garment_analysis: Dict,
                                 product_info: Dict, original_image: np.ndarray) -> Dict:
        """Debug function to identify pipeline issues - ADD THIS FOR DEBUGGING"""
        
        debug_info = {}
        
        # 1. Check customer analysis
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        debug_info['landmarks_available'] = list(pose_landmarks.keys())
        debug_info['landmark_count'] = len(pose_landmarks)
        
        # 2. Check garment analysis
        dominant_colors = garment_analysis.get('dominant_colors', [])
        debug_info['dominant_colors'] = dominant_colors
        debug_info['color_count'] = len(dominant_colors)
        
        # 3. Check product info
        product_name = product_info.get('name', '')
        debug_info['product_name'] = product_name
        debug_info['name_contains_white'] = 'white' in product_name.lower()
        
        # 4. Check image properties
        debug_info['image_shape'] = original_image.shape
        debug_info['image_size'] = original_image.size
        debug_info['image_dtype'] = str(original_image.dtype)
        
        # 5. Test mask creation
        try:
            test_mask = self._create_region_mask(GarmentType.TOP, customer_analysis, original_image.shape)
            if test_mask is not None:
                debug_info['mask_created'] = True
                debug_info['mask_area'] = int(np.sum(test_mask > 128))
                debug_info['mask_coverage_percent'] = float((np.sum(test_mask > 128) / test_mask.size) * 100)
            else:
                debug_info['mask_created'] = False
        except Exception as e:
            debug_info['mask_error'] = str(e)
            debug_info['mask_created'] = False
        
        self.logger.info(f"[DEBUG] Pipeline analysis: {debug_info}")
        return debug_info


# Integration function for your existing system
def process_comprehensive_tryon(customer_analysis: Dict, garment_analysis: Dict,
                              product_info: Dict, original_image: np.ndarray,
                              garment_types: List[str]) -> ProcessingResult:
    """
    Main integration function to replace your SAFE mode
    This is what you'll call from your existing code
    """
    
    processor = ComprehensiveRegionTryOn()
    return processor.process_virtual_tryon(
        customer_analysis=customer_analysis,
        garment_analysis=garment_analysis,
        product_info=product_info,
        original_image=original_image,
        garment_types=garment_types
    )