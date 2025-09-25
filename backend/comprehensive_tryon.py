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
    
    def _create_top_mask(self, pose_landmarks: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create mask for tops (shirts, t-shirts, blouses)"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get landmarks
        ls = pose_landmarks.get('left_shoulder', {})
        rs = pose_landmarks.get('right_shoulder', {})
        lh = pose_landmarks.get('left_hip', {})
        rh = pose_landmarks.get('right_hip', {})
        
        if all(point.get('confidence', 0) > 0.7 for point in [ls, rs, lh, rh]):
            # Convert to pixels
            ls_x, ls_y = int(ls['x'] * width), int(ls['y'] * height)
            rs_x, rs_y = int(rs['x'] * width), int(rs['y'] * height)
            lh_x, lh_y = int(lh['x'] * width), int(lh['y'] * height)
            rh_x, rh_y = int(rh['x'] * width), int(rh['y'] * height)
            
            # Create torso polygon excluding arms
            shoulder_width = abs(rs_x - ls_x)
            torso_width = int(shoulder_width * 0.75)  # 75% to exclude arms
            center_x = (ls_x + rs_x) // 2
            
            # Define torso boundaries
            top_y = min(ls_y, rs_y) - 10  # Slightly above shoulders
            bottom_y = max(lh_y, rh_y) + 10  # Slightly below hips
            left_x = center_x - torso_width // 2
            right_x = center_x + torso_width // 2
            
            # Create rounded rectangle for natural torso shape
            points = np.array([
                [left_x, top_y + 20],       # Top left (rounded)
                [right_x, top_y + 20],      # Top right (rounded)
                [right_x, bottom_y],        # Bottom right
                [lh_x + 20, bottom_y],      # Right hip
                [lh_x - 20, bottom_y],      # Left hip  
                [left_x, bottom_y],         # Bottom left
            ], dtype=np.int32)
            
            cv2.fillPoly(mask, [points], 255)
            
            # Smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (11, 11), 4)
            
            self.logger.info(f"[TOP-MASK] Created torso mask: {np.sum(mask > 0)} pixels")
            return mask
            
        return None
    
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
        
        # Determine base color
        product_name = product_info.get('name', '').lower()
        if 'white' in product_name:
            base_color = (255, 255, 255)
        elif 'black' in product_name:
            base_color = (0, 0, 0)
        elif 'blue' in product_name:
            base_color = (100, 50, 200)
        else:
            base_color = dominant_colors[0] if dominant_colors else (128, 128, 128)
        
        # Fill base color
        garment_region[:] = base_color
        
        # Add garment-specific effects
        roughness = texture_features.get('roughness', 0.3)
        complexity = texture_features.get('complexity', 0.1)
        
        # Add fabric texture
        if roughness > 0.2:
            noise_strength = int(roughness * 15)
            noise = np.random.normal(0, noise_strength, garment_region.shape)
            garment_region = np.clip(
                garment_region.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
        
        # Add patterns or secondary colors if detected
        if len(dominant_colors) > 1 and complexity > 0.05:
            self._add_garment_patterns(garment_region, dominant_colors[1:], complexity)
        
        # Add garment-specific details
        if garment_type == GarmentType.SHOES:
            garment_region = self._add_shoe_details(garment_region, mask)
        elif garment_type == GarmentType.BOTTOM:
            garment_region = self._add_pants_details(garment_region, mask)
        
        self.logger.info(f"[APPEARANCE] Generated {garment_type.value} with base color {base_color}")
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
        """Seamlessly blend garment region with existing image"""
        
        # Create smooth transition mask
        smooth_mask = cv2.GaussianBlur(mask, (19, 19), 6).astype(np.float32) / 255.0
        
        # Blend garment with base image
        result = base_image.copy().astype(np.float32)
        garment_float = garment_region.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (
                base_image[:, :, c].astype(np.float32) * (1 - smooth_mask) +
                garment_float[:, :, c] * smooth_mask
            )
        
        # Apply edge enhancement for seamless integration
        result = self._enhance_blend_edges(result.astype(np.uint8), base_image, mask)
        
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
        """Calculate overall quality score for the result"""
        scores = []
        
        # Image quality score
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(result_gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 1000.0)
        scores.append(sharpness_score)
        
        # Color consistency score
        result_std = np.std(result)
        original_std = np.std(original)
        if original_std > 0:
            color_consistency = 1.0 - abs(result_std - original_std) / original_std
            scores.append(max(0.5, color_consistency))
        
        # Region modification score (penalize excessive modification)
        modification_penalty = max(0.7, 1.0 - len(modified_regions) * 0.1)
        scores.append(modification_penalty)
        
        return float(np.mean(scores))


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