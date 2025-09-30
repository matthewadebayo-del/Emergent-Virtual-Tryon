import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

class PracticalGarmentReplacer:
    """
    Practical solution for complete garment replacement
    This replaces your current garment replacement with Mask Expansion of 60% and more Debug Logging that is Easier for troubleshooting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def replace_garment_completely(self, customer_analysis: Dict, garment_analysis: Dict,
                                 product_info: Dict, original_image: np.ndarray,
                                 garment_types: List[str]) -> np.ndarray:
        """
        REALISTIC garment replacement - uses advanced rendering system
        """
        
        print("DEBUG: replace_garment_completely METHOD CALLED - USING REALISTIC RENDERING")
        
        # Call the realistic garment replacement system directly
        return replace_garment_with_realism(
            customer_analysis=customer_analysis,
            garment_analysis=garment_analysis,
            product_info=product_info,
            original_image=original_image,
            garment_types=garment_types
        )
    
    def _force_correct_color(self, product_info: Dict, garment_analysis: Dict) -> Tuple[int, int, int]:
        """
        FORCE correct color based on product name - no more gray for white shirts!
        """
        
        product_name = product_info.get('name', '').lower()
        self.logger.info(f"[COLOR] Product name: '{product_name}'")
        
        # ABSOLUTE priority to product name
        if any(word in product_name for word in ['white', 'blanc', 'blanco', 'wit']):
            color = (255, 255, 255)  # PURE WHITE
            self.logger.info(f"[COLOR] FORCING WHITE: {color}")
            return color
            
        elif any(word in product_name for word in ['black', 'noir', 'negro']):
            color = (10, 10, 10)     # PURE BLACK
            self.logger.info(f"[COLOR] FORCING BLACK: {color}")
            return color
            
        elif any(word in product_name for word in ['red', 'rouge', 'rojo']):
            color = (220, 20, 20)    # BRIGHT RED
            self.logger.info(f"[COLOR] FORCING RED: {color}")
            return color
            
        elif any(word in product_name for word in ['blue', 'bleu', 'azul']):
            color = (20, 20, 220)    # BRIGHT BLUE
            self.logger.info(f"[COLOR] FORCING BLUE: {color}")
            return color
            
        else:
            # Check if analyzed color is actually white-ish
            dominant_colors = garment_analysis.get('dominant_colors', [])
            if dominant_colors:
                # Convert numpy arrays to regular tuples
                color_data = dominant_colors[0]
                if hasattr(color_data, 'tolist'):
                    r, g, b = color_data.tolist()[:3]
                else:
                    r, g, b = color_data[:3]
                
                # Convert to int to avoid numpy scalar issues
                r, g, b = int(r), int(g), int(b)
                
                # If any component > 180, force pure white
                if r > 180 and g > 180 and b > 180:
                    color = (255, 255, 255)
                    self.logger.info(f"[COLOR] Light color ({r}, {g}, {b}) -> FORCING WHITE")
                    return color
                else:
                    color = (r, g, b)
                    self.logger.info(f"[COLOR] Using analyzed color: {color}")
                    return color
            else:
                color = (128, 128, 128)
                self.logger.warning(f"[COLOR] Fallback to gray: {color}")
                return color
    
    def _create_complete_removal_mask(self, customer_analysis: Dict, image_shape: Tuple) -> np.ndarray:
        """
        Create BALANCED mask that covers the shirt area with 10-20% coverage for realistic replacement
        """
        
        height, width, _ = image_shape
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        
        # Get landmark positions
        landmarks = {}
        for lm_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if lm_name in pose_landmarks:
                lm = pose_landmarks[lm_name]
                # Handle both dict format {'x': 0.5, 'y': 0.3} and list format [0.5, 0.3]
                if isinstance(lm, dict):
                    x = int(lm.get('x', 0) * width)
                    y = int(lm.get('y', 0) * height)
                else:
                    x = int(lm[0] * width)
                    y = int(lm[1] * height)
                landmarks[lm_name] = (x, y)
                print(f"LANDMARK {lm_name}: ({x}, {y})")
        
        if len(landmarks) < 4:
            print(f"ERROR: Insufficient landmarks: {list(landmarks.keys())} out of 4 required")
            self.logger.error("[MASK] Insufficient landmarks for removal mask")
            return np.zeros((height, width), dtype=np.uint8)
        
        # Calculate dimensions
        shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        torso_height = abs(landmarks['left_hip'][1] - landmarks['left_shoulder'][1])
        
        print(f"DIMENSIONS: Shoulder width={shoulder_width}, Torso height={torso_height}")
        
        # BALANCED expansion - target 10-20% coverage
        horizontal_expansion = max(35, int(shoulder_width * 0.25))  # 25% wider than shoulders
        vertical_expansion = max(25, int(torso_height * 0.15))      # 15% beyond torso height
        
        print(f"EXPANSION: Horizontal={horizontal_expansion}, Vertical={vertical_expansion}")
        
        # Create shirt-covering polygon
        ls = landmarks['left_shoulder']
        rs = landmarks['right_shoulder']
        lh = landmarks['left_hip']
        rh = landmarks['right_hip']
        
        print(f"POLYGON: ls={ls}, rs={rs}, lh={lh}, rh={rh}")
        
        # Find actual left and right based on x coordinates
        if ls[0] > rs[0]:  # left_shoulder is actually on the right side
            actual_left = rs  # right_shoulder is actually on left
            actual_right = ls  # left_shoulder is actually on right
            actual_left_hip = rh
            actual_right_hip = lh
        else:
            actual_left = ls
            actual_right = rs
            actual_left_hip = lh
            actual_right_hip = rh
        
        print(f"CORRECTED: left={actual_left}, right={actual_right}")
        
        # Create proper shirt-covering polygon
        polygon_points = np.array([
            # Top-left (expanded outward and upward)
            (actual_left[0] - horizontal_expansion, actual_left[1] - vertical_expansion),
            # Top-right (expanded outward and upward)
            (actual_right[0] + horizontal_expansion, actual_right[1] - vertical_expansion),
            # Mid-right (slight outward curve)
            (actual_right[0] + horizontal_expansion, (actual_right[1] + actual_right_hip[1]) // 2),
            # Bottom-right (expanded downward)
            (actual_right_hip[0] + horizontal_expansion//2, actual_right_hip[1] + vertical_expansion),
            # Bottom-left (expanded downward)
            (actual_left_hip[0] - horizontal_expansion//2, actual_left_hip[1] + vertical_expansion),
            # Mid-left (slight outward curve)
            (actual_left[0] - horizontal_expansion, (actual_left[1] + actual_left_hip[1]) // 2),
        ], dtype=np.int32)
        
        print(f"POLYGON BEFORE CLIP: {polygon_points}")
        
        # Ensure within bounds
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, width - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, height - 1)
        
        print(f"POLYGON AFTER CLIP: {polygon_points}")
        print(f"IMAGE BOUNDS: width={width}, height={height}")
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        print(f"MASK SHAPE: {mask.shape}")
        
        try:
            cv2.fillPoly(mask, [polygon_points], 255)
            print(f"FILLPOLY SUCCESS")
        except Exception as e:
            print(f"FILLPOLY ERROR: {e}")
            return np.zeros((height, width), dtype=np.uint8)
        
        # Light smoothing to preserve mask area
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
        # Log mask statistics
        mask_area = np.sum(mask > 128)
        coverage = (mask_area / (width * height)) * 100
        print(f"FINAL MASK: {mask_area} pixels ({coverage:.1f}% coverage)")
        
        # Validate coverage is in target range
        if coverage < 8:
            print(f"WARNING: Mask still too small ({coverage:.1f}%)")
            # Apply dilation to increase size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask_area = np.sum(mask > 128)
            coverage = (mask_area / (width * height)) * 100
            print(f"EXPANDED MASK: {mask_area} pixels ({coverage:.1f}% coverage)")
        elif coverage > 30:
            print(f"WARNING: Mask too large ({coverage:.1f}%)")
            # Apply erosion to decrease size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask_area = np.sum(mask > 128)
            coverage = (mask_area / (width * height)) * 100
            print(f"REDUCED MASK: {mask_area} pixels ({coverage:.1f}% coverage)")
        else:
            print(f"SUCCESS: Good mask size ({coverage:.1f}%)")
        
        return mask
    
    def _create_fallback_mask(self, height: int, width: int) -> np.ndarray:
        """Create fallback mask if landmarks are insufficient"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Cover central torso area
        center_x, center_y = width // 2, height // 2
        torso_width = width // 3
        torso_height = height // 2
        
        cv2.rectangle(mask, 
                     (center_x - torso_width//2, center_y - torso_height//3),
                     (center_x + torso_width//2, center_y + torso_height//2),
                     255, -1)
        
        mask = cv2.GaussianBlur(mask, (41, 41), 20)
        self.logger.warning(f"[MASK] Using fallback central mask")
        return mask
    
    def _remove_original_garment_completely(self, original_image: np.ndarray, 
                                          removal_mask: np.ndarray) -> np.ndarray:
        """
        Remove original garment with VISIBLE removal - creates obvious "shirt removed" effect
        """
        
        print("REMOVAL: Starting visible garment removal...")
        
        # Step 1: Estimate skin tone from non-garment areas
        skin_tone = self._estimate_skin_tone_better(original_image, removal_mask)
        print(f"REMOVAL: Estimated skin tone: {skin_tone}")
        
        # Step 2: Create "body without shirt" by replacing garment area with skin
        body_without_shirt = original_image.copy()
        
        # Create strong removal mask (core area gets complete replacement)
        core_removal_mask = (removal_mask > 200).astype(np.float32)
        edge_removal_mask = cv2.GaussianBlur(removal_mask, (21, 21), 10).astype(np.float32) / 255.0
        
        # Combine for strong core removal with smooth edges
        final_removal_mask = np.maximum(core_removal_mask * 0.95, edge_removal_mask * 0.7)
        
        # Step 3: Apply skin tone replacement
        garment_area = removal_mask > 100
        for c in range(3):
            body_without_shirt[:, :, c][garment_area] = (
                original_image[:, :, c][garment_area] * (1 - final_removal_mask[garment_area]) +
                skin_tone[c] * final_removal_mask[garment_area]
            )
        
        # Step 4: Add subtle body contours/shadows to make it look like bare skin
        body_without_shirt = self._add_body_contours(body_without_shirt, removal_mask, skin_tone)
        
        # Step 5: Verify removal was visible
        removal_diff = np.sum(cv2.absdiff(body_without_shirt, original_image))
        removal_area = np.sum(removal_mask > 100)
        avg_change_per_pixel = removal_diff / (removal_area * 3) if removal_area > 0 else 0
        
        print(f"REMOVAL: Total difference: {removal_diff}")
        print(f"REMOVAL: Avg change per pixel: {avg_change_per_pixel:.1f}")
        
        if avg_change_per_pixel < 20:
            print("WARNING: Garment removal barely visible!")
        else:
            print("SUCCESS: Visible garment removal achieved")
        
        return body_without_shirt
    
    def _estimate_skin_tone(self, image: np.ndarray, garment_mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Estimate skin tone from non-garment areas (face, arms, etc.)
        """
        
        try:
            height, width = image.shape[:2]
            
            # Create skin sampling mask (exclude garment area)
            skin_mask = np.ones((height, width), dtype=np.uint8) * 255
            skin_mask[garment_mask > 100] = 0  # Exclude garment
            
            # Focus on upper areas (likely face/neck)
            skin_mask[height//2:] = 0  # Only upper half
            skin_mask[:height//4] = 0  # Exclude very top (might be hair)
            
            # Sample skin pixels
            skin_pixels = image[skin_mask > 0]
            
            if len(skin_pixels) > 100:  # Need sufficient samples
                # Use median for robustness against outliers - fix numpy scalar issue
                median_values = np.median(skin_pixels, axis=0)
                if hasattr(median_values, 'tolist'):
                    skin_tone = tuple(int(x) for x in median_values.tolist())
                else:
                    skin_tone = tuple(int(x) for x in median_values)
            else:
                # Fallback to a neutral skin tone
                skin_tone = (200, 170, 150)
            
            self.logger.info(f"[SKIN] Estimated skin tone: {skin_tone} from {len(skin_pixels)} pixels")
            return skin_tone
            
        except Exception as e:
            self.logger.error(f"[SKIN] Skin tone estimation failed: {e}")
            return (200, 170, 150)  # Safe fallback
    
    def _estimate_skin_tone_better(self, image: np.ndarray, garment_mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Better skin tone estimation from multiple areas
        """
        
        height, width = image.shape[:2]
        
        # Sample from multiple non-garment areas
        skin_samples = []
        
        # Area 1: Upper face/neck (exclude very top which might be hair)
        face_mask = np.zeros((height, width), dtype=np.uint8)
        face_mask[height//8:height//3, width//4:3*width//4] = 255
        face_mask[garment_mask > 50] = 0  # Exclude garment area
        
        face_pixels = image[face_mask > 0]
        if len(face_pixels) > 100:
            skin_samples.extend(face_pixels)
        
        # Area 2: Arms (if visible)
        arms_mask = np.zeros((height, width), dtype=np.uint8)
        arms_mask[height//3:2*height//3, :width//5] = 255  # Left side
        arms_mask[height//3:2*height//3, 4*width//5:] = 255  # Right side
        arms_mask[garment_mask > 50] = 0
        
        arm_pixels = image[arms_mask > 0]
        if len(arm_pixels) > 50:
            skin_samples.extend(arm_pixels)
        
        # Calculate robust skin tone
        if len(skin_samples) > 200:
            skin_samples = np.array(skin_samples)
            # Use median of middle 50% to avoid outliers
            skin_tone = []
            for c in range(3):
                channel_values = skin_samples[:, c]
                p25, p75 = np.percentile(channel_values, [25, 75])
                middle_values = channel_values[(channel_values >= p25) & (channel_values <= p75)]
                skin_tone.append(int(np.median(middle_values)))
            skin_tone = tuple(skin_tone)
        else:
            # Fallback: analyze the image border (often contains skin)
            border_pixels = np.concatenate([
                image[0, :].reshape(-1, 3),        # Top border
                image[-1, :].reshape(-1, 3),       # Bottom border  
                image[:, 0].reshape(-1, 3),        # Left border
                image[:, -1].reshape(-1, 3)        # Right border
            ])
            
            if len(border_pixels) > 0:
                # Filter out very dark or very light pixels (likely clothing/background)
                valid_pixels = border_pixels[
                    (np.mean(border_pixels, axis=1) > 50) & 
                    (np.mean(border_pixels, axis=1) < 200)
                ]
                if len(valid_pixels) > 0:
                    skin_tone = tuple(int(x) for x in np.median(valid_pixels, axis=0))
                else:
                    skin_tone = (180, 150, 120)  # Generic skin tone
            else:
                skin_tone = (180, 150, 120)
        
        print(f"SKIN: Analyzed {len(skin_samples)} skin pixels")
        return skin_tone
    
    def _add_body_contours(self, body_image: np.ndarray, mask: np.ndarray, 
                          skin_tone: Tuple[int, int, int]) -> np.ndarray:
        """
        Add subtle body contours to make removed area look like real skin/body
        """
        
        result = body_image.copy()
        garment_area = mask > 100
        
        if not np.any(garment_area):
            return result
        
        # Create subtle shadow gradients to simulate body curvature
        height, width = body_image.shape[:2]
        
        # Find the center of the garment area
        y_coords, x_coords = np.where(garment_area)
        if len(y_coords) == 0:
            return result
            
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Create radial gradient from center (lighter) to edges (darker)
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distances within garment area
        garment_distances = distances[garment_area]
        if len(garment_distances) > 0:
            max_dist = np.max(garment_distances)
            min_dist = np.min(garment_distances)
            
            if max_dist > min_dist:
                # Create subtle shading (5-15% variation)
                normalized_dist = (distances - min_dist) / (max_dist - min_dist)
                shading_factor = 0.95 + normalized_dist * 0.1  # 0.95 to 1.05 range
                shading_factor = np.clip(shading_factor, 0.85, 1.15)
                
                # Apply shading only to garment area
                for c in range(3):
                    result[:, :, c][garment_area] = np.clip(
                        result[:, :, c][garment_area] * shading_factor[garment_area], 0, 255
                    )
        
        # Add very subtle texture to make it look like skin rather than flat color
        garment_area_coords = np.where(garment_area)
        if len(garment_area_coords[0]) > 0:
            # Add minimal noise for skin texture
            noise = np.random.normal(0, 2, (len(garment_area_coords[0]), 3))
            for i, (y, x) in enumerate(zip(garment_area_coords[0], garment_area_coords[1])):
                result[y, x] = np.clip(result[y, x] + noise[i], 0, 255)
        
        print("CONTOURS: Added body contours and skin texture")
        return result
    

    
    def debug_visual_stages(self, original_image, removal_mask, customer_analysis, garment_analysis, product_info):
        """Debug function to visualize each transformation stage"""
        
        # Stage 1: Remove original garment
        body_without_shirt = self._remove_original_garment_completely(original_image, removal_mask)
        
        # Save for inspection
        try:
            cv2.imwrite('/tmp/stage1_removal.jpg', body_without_shirt)
        except:
            pass  # Skip if can't write to /tmp
        removal_diff = np.sum(cv2.absdiff(body_without_shirt, original_image))
        print(f"STAGE 1: Removal created {removal_diff} pixel difference")
        
        # Stage 2: Create new garment
        new_garment = self._create_new_garment(
            self._force_correct_color(product_info, garment_analysis), 
            removal_mask, 
            original_image.shape
        )
        
        # Save garment for inspection
        try:
            cv2.imwrite('/tmp/stage2_new_garment.jpg', new_garment)
        except:
            pass
        
        # Stage 3: Apply new garment
        final_result = self._apply_new_garment_strongly(body_without_shirt, new_garment, removal_mask)
        
        # Save final result
        try:
            cv2.imwrite('/tmp/stage3_final.jpg', final_result)
        except:
            pass
        final_diff = np.sum(cv2.absdiff(final_result, original_image))
        print(f"STAGE 3: Final transformation: {final_diff} pixel difference")
        
        return final_result
    
    def _generate_realistic_garment_advanced(self, garment_analysis: Dict, product_info: Dict,
                                           image_shape: Tuple, mask: np.ndarray, 
                                           original_image: np.ndarray) -> np.ndarray:
        """
        Generate realistic garment that looks like actual clothing, not flat color
        """
        
        height, width, _ = image_shape
        garment = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get base color (existing logic)
        product_name = product_info.get('name', '').lower()
        if any(color_word in product_name for color_word in ['white', 'blanc', 'blanco']):
            base_color = (255, 255, 255)
            print("GARMENT: Using WHITE")
        elif any(color_word in product_name for color_word in ['black', 'noir', 'negro']):
            base_color = (20, 20, 20)
            print("GARMENT: Using BLACK")
        else:
            dominant_colors = garment_analysis.get('dominant_colors', [])
            base_color = dominant_colors[0] if dominant_colors else (128, 128, 128)
            print(f"GARMENT: Using {base_color}")
        
        # Step 1: Create realistic garment shape (not rectangular)
        realistic_garment_mask = self._create_realistic_garment_shape(mask, original_image)
        
        # Step 2: Apply base color with lighting variation
        garment = self._apply_realistic_lighting(garment, realistic_garment_mask, base_color, original_image)
        
        # Step 3: Add fabric texture and details
        garment = self._add_fabric_realism(garment, realistic_garment_mask, base_color)
        
        # Step 4: Add garment structure (seams, folds, etc.)
        garment = self._add_garment_structure(garment, realistic_garment_mask, base_color)
        
        print(f"GARMENT: Generated realistic garment")
        return garment
    
    def _create_realistic_garment_shape(self, base_mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Create realistic t-shirt shape that follows body contours
        """
        
        height, width = base_mask.shape
        
        # Start with base mask but make it more shirt-like
        shirt_mask = base_mask.copy()
        
        # Find the garment region
        garment_coords = np.where(shirt_mask > 100)
        if len(garment_coords[0]) == 0:
            return shirt_mask
        
        # Get bounding box of garment area
        min_y, max_y = np.min(garment_coords[0]), np.max(garment_coords[0])
        min_x, max_x = np.min(garment_coords[1]), np.max(garment_coords[1])
        
        # Create more realistic t-shirt shape
        center_x = (min_x + max_x) // 2
        garment_width = max_x - min_x
        garment_height = max_y - min_y
        
        # Clear the mask and redraw with realistic proportions
        shirt_mask = np.zeros_like(shirt_mask)
        
        # Create t-shirt shape points
        shoulder_width = int(garment_width * 0.85)  # Slightly narrower than full width
        bottom_width = int(garment_width * 0.75)    # Tapered at bottom
        
        # Define t-shirt contour points
        shirt_points = [
            # Shoulders and neckline
            (center_x - shoulder_width//2, min_y),
            (center_x + shoulder_width//2, min_y),
            
            # Armpit area (slight curve inward)
            (center_x + shoulder_width//2 - 5, min_y + garment_height//4),
            (center_x + bottom_width//2, min_y + garment_height//2),
            
            # Bottom hem
            (center_x + bottom_width//2, max_y),
            (center_x - bottom_width//2, max_y),
            
            # Left side back to shoulder
            (center_x - bottom_width//2, min_y + garment_height//2),
            (center_x - shoulder_width//2 + 5, min_y + garment_height//4),
        ]
        
        # Convert to numpy array and ensure within bounds
        shirt_points = np.array(shirt_points, dtype=np.int32)
        shirt_points[:, 0] = np.clip(shirt_points[:, 0], 0, width - 1)
        shirt_points[:, 1] = np.clip(shirt_points[:, 1], 0, height - 1)
        
        # Fill the realistic shirt shape
        cv2.fillPoly(shirt_mask, [shirt_points], 255)
        
        # Smooth the shape
        shirt_mask = cv2.GaussianBlur(shirt_mask, (15, 15), 6)
        
        print(f"SHAPE: Created realistic t-shirt shape")
        return shirt_mask
    
    def _apply_realistic_lighting(self, garment: np.ndarray, mask: np.ndarray, 
                                base_color: Tuple[int, int, int], original_image: np.ndarray) -> np.ndarray:
        """
        Apply realistic lighting and shadows based on the original scene
        """
        
        # Extract lighting from original image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        lighting_map = cv2.GaussianBlur(original_gray, (71, 71), 30).astype(np.float32) / 255.0
        
        # Create base lighting variation (not flat color)
        garment_area = mask > 100
        
        for c in range(3):
            # Apply scene lighting
            lit_color = base_color[c] * lighting_map
            
            # Add subtle variation to avoid flat appearance
            variation = np.random.normal(1.0, 0.02, lighting_map.shape)  # 2% variation
            lit_color *= variation
            
            # Apply to garment
            garment[:, :, c][garment_area] = np.clip(lit_color[garment_area], 0, 255)
        
        print(f"LIGHTING: Applied realistic scene lighting")
        return garment
    
    def _add_fabric_realism(self, garment: np.ndarray, mask: np.ndarray, 
                           base_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Add realistic fabric texture and appearance
        """
        
        garment_area = mask > 100
        garment_coords = np.where(garment_area)
        
        if len(garment_coords[0]) == 0:
            return garment
        
        # Add fabric texture based on garment color
        if base_color == (255, 255, 255):  # White fabric
            # Very subtle texture for white cotton
            texture_strength = 2
            fabric_pattern = np.random.normal(0, texture_strength, (len(garment_coords[0]), 3))
            
            # Add slight fabric weave pattern
            for i, (y, x) in enumerate(zip(garment_coords[0], garment_coords[1])):
                weave_factor = 1 + 0.01 * np.sin(x * 0.2) * np.cos(y * 0.15)  # Subtle weave
                garment[y, x] = np.clip(garment[y, x] * weave_factor + fabric_pattern[i], 240, 255)
                
        else:  # Colored fabric
            texture_strength = 4
            fabric_pattern = np.random.normal(0, texture_strength, (len(garment_coords[0]), 3))
            
            for i, (y, x) in enumerate(zip(garment_coords[0], garment_coords[1])):
                garment[y, x] = np.clip(garment[y, x] + fabric_pattern[i], 0, 255)
        
        print(f"FABRIC: Added realistic fabric texture")
        return garment
    
    def _add_garment_structure(self, garment: np.ndarray, mask: np.ndarray, 
                             base_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Add realistic garment structure: seams, folds, natural draping
        """
        
        # Find garment boundaries
        garment_coords = np.where(mask > 100)
        if len(garment_coords[0]) == 0:
            return garment
        
        min_y, max_y = np.min(garment_coords[0]), np.max(garment_coords[0])
        min_x, max_x = np.min(garment_coords[1]), np.max(garment_coords[1])
        center_x = (min_x + max_x) // 2
        
        # Create slightly darker color for seams and structure
        structure_color = tuple(max(0, int(c * 0.9)) for c in base_color)
        
        # Add shoulder seams (subtle horizontal lines)
        shoulder_y = min_y + (max_y - min_y) // 6
        cv2.line(garment, 
                 (center_x - 30, shoulder_y), 
                 (center_x + 30, shoulder_y), 
                 structure_color, 1)
        
        # Add side seams (very subtle vertical lines)
        left_seam_x = center_x - (max_x - min_x) // 4
        right_seam_x = center_x + (max_x - min_x) // 4
        seam_start_y = min_y + (max_y - min_y) // 4
        seam_end_y = min_y + 3 * (max_y - min_y) // 4
        
        cv2.line(garment, (left_seam_x, seam_start_y), (left_seam_x, seam_end_y), structure_color, 1)
        cv2.line(garment, (right_seam_x, seam_start_y), (right_seam_x, seam_end_y), structure_color, 1)
        
        # Add subtle bottom hem
        hem_y = min_y + 4 * (max_y - min_y) // 5
        cv2.line(garment, 
                 (center_x - (max_x - min_x) // 3, hem_y), 
                 (center_x + (max_x - min_x) // 3, hem_y), 
                 structure_color, 1)
        
        # Add very subtle body-following contours
        garment = self._add_body_following_contours(garment, mask, base_color)
        
        print(f"STRUCTURE: Added garment seams and structure")
        return garment
    
    def _add_body_following_contours(self, garment: np.ndarray, mask: np.ndarray, 
                                    base_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Add subtle contours that make the garment appear to follow body shape
        """
        
        garment_area = mask > 100
        garment_coords = np.where(garment_area)
        
        if len(garment_coords[0]) == 0:
            return garment
        
        # Find center of garment for body curvature simulation
        center_y = int(np.mean(garment_coords[0]))
        center_x = int(np.mean(garment_coords[1]))
        
        # Create subtle radial shading from center (lighter) to edges (slightly darker)
        height, width = garment.shape[:2]
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distances within garment area
        garment_distances = distances[garment_area]
        if len(garment_distances) > 0:
            max_dist = np.max(garment_distances)
            
            if max_dist > 0:
                # Very subtle shading (2-5% variation)
                normalized_dist = distances / max_dist
                shading_factor = 1.0 - normalized_dist * 0.03  # Very subtle
                shading_factor = np.clip(shading_factor, 0.97, 1.0)
                
                # Apply only to garment area
                for c in range(3):
                    garment[:, :, c][garment_area] = np.clip(
                        garment[:, :, c][garment_area] * shading_factor[garment_area], 0, 255
                    )
        
        print(f"CONTOURS: Added body-following contours")
        return garment
    
    def _apply_new_garment_naturally(self, body_image: np.ndarray, realistic_garment: np.ndarray, 
                                    mask: np.ndarray) -> np.ndarray:
        """
        Apply new garment with natural blending that looks like real clothing
        """
        
        # Create natural blending mask (stronger in center, softer at edges)
        core_mask = (mask > 220).astype(np.float32) * 0.92  # 92% in core
        
        # Create graduated edge blending
        edge_mask = cv2.GaussianBlur(mask, (25, 25), 12).astype(np.float32) / 255.0
        edge_mask = edge_mask * 0.85  # 85% at edges
        
        # Combine for natural transition
        final_mask = np.maximum(core_mask, edge_mask)
        
        # Ensure smooth transitions at garment boundaries
        final_mask = cv2.GaussianBlur(final_mask, (7, 7), 2)
        
        # Apply the realistic garment
        result = body_image.copy().astype(np.float32)
        garment_float = realistic_garment.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (
                body_image[:, :, c] * (1 - final_mask) +
                garment_float[:, :, c] * final_mask
            )
        
        # Add slight color temperature matching
        result = self._match_scene_colors(result.astype(np.uint8), body_image, mask)
        
        print(f"APPLICATION: Applied garment naturally")
        return result
    
    def _match_scene_colors(self, result: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Subtle color temperature matching to scene
        """
        
        # Sample non-garment areas for scene color temperature
        non_garment = mask < 50
        garment_area = mask > 100
        
        if np.sum(non_garment) > 1000 and np.sum(garment_area) > 1000:
            # Get average color temperature of scene
            scene_avg = np.mean(original[non_garment], axis=0)
            
            # Very subtle color temperature adjustment (only 5% influence)
            adjustment_factor = 0.05
            
            for c in range(3):
                if scene_avg[c] > 0:
                    # Subtle warm/cool adjustment based on scene
                    temp_factor = 1.0 + (scene_avg[c] / 255.0 - 0.5) * adjustment_factor
                    result[:, :, c][garment_area] = np.clip(
                        result[:, :, c][garment_area] * temp_factor, 0, 255
                    )
        
        return result
    

    
    def _apply_garment_with_strong_replacement(self, body_image: np.ndarray, 
                                             new_garment: np.ndarray, mask: np.ndarray,
                                             original_image: np.ndarray) -> np.ndarray:
        """
        Apply new garment with STRONG replacement (not subtle blending)
        """
        
        # Create strong replacement mask
        # Core area gets 95% replacement, edges blend smoothly
        core_mask = (mask > 220).astype(np.float32) * 0.95
        edge_mask = cv2.GaussianBlur(mask, (31, 31), 15).astype(np.float32) / 255.0
        
        # Combine masks for strong core replacement with smooth edges
        final_mask = np.maximum(core_mask, edge_mask * 0.7)
        
        # Apply lighting from original image to new garment
        lighting_adjusted_garment = self._apply_scene_lighting(new_garment, original_image, mask)
        
        # Strong compositing
        result = body_image.copy().astype(np.float32)
        garment_float = lighting_adjusted_garment.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (
                body_image[:, :, c] * (1 - final_mask) +
                garment_float[:, :, c] * final_mask
            )
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Log replacement statistics
        replacement_area = np.sum(final_mask > 0.5)
        self.logger.info(f"[REPLACE] Strong replacement applied to {replacement_area} pixels")
        
        return result
    
    def _apply_scene_lighting(self, garment: np.ndarray, original_image: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        Apply lighting from the original scene to the new garment
        """
        
        # Extract lighting information from original image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        lighting_map = cv2.GaussianBlur(original_gray, (71, 71), 35).astype(np.float32) / 255.0
        
        # Apply lighting to garment
        lit_garment = garment.copy().astype(np.float32)
        
        # Only apply lighting in garment area
        garment_area = mask > 100
        for c in range(3):
            lit_garment[:, :, c][garment_area] *= lighting_map[garment_area]
        
        return np.clip(lit_garment, 0, 255).astype(np.uint8)
    
    def _apply_final_realism_adjustments(self, result: np.ndarray, original_image: np.ndarray,
                                       mask: np.ndarray) -> np.ndarray:
        """
        Apply final adjustments for maximum realism
        """
        
        # Color temperature matching
        result = self._match_color_temperature(result, original_image, mask)
        
        # Subtle sharpening
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
        
        return result
    
    def _match_color_temperature(self, result: np.ndarray, original: np.ndarray,
                               mask: np.ndarray) -> np.ndarray:
        """
        Match color temperature of the scene
        """
        
        # Sample non-garment areas for color temperature
        non_garment = mask < 50
        if np.sum(non_garment) > 1000:
            original_temp = np.mean(original[non_garment], axis=0)
            result_temp = np.mean(result[non_garment], axis=0)
            
            # Calculate subtle color correction
            temp_ratio = original_temp / (result_temp + 1e-6)
            
            # Apply gentle correction only to garment area
            garment_area = mask > 100
            for c in range(3):
                result[:, :, c][garment_area] = np.clip(
                    result[:, :, c][garment_area] * (temp_ratio[c] * 0.1 + 0.9), 0, 255
                )
        
        return result

# Complete realistic garment replacement function
def replace_garment_with_realism(customer_analysis: Dict, garment_analysis: Dict,
                               product_info: Dict, original_image: np.ndarray,
                               garment_types: List[str]) -> np.ndarray:
    """
    Complete garment replacement with realistic rendering
    """
    
    print("STARTING: Realistic garment replacement...")
    
    # Create mask (existing method)
    processor = PracticalGarmentReplacer()
    removal_mask = processor._create_complete_removal_mask(
        customer_analysis, original_image.shape
    )
    
    # Remove original garment (existing method) 
    body_without_garment = processor._remove_original_garment_completely(original_image, removal_mask)
    
    # Generate REALISTIC garment (new method)
    realistic_garment = processor._generate_realistic_garment_advanced(
        garment_analysis, product_info, original_image.shape, removal_mask, original_image
    )
    
    # Apply with natural blending (new method)
    final_result = processor._apply_new_garment_naturally(body_without_garment, realistic_garment, removal_mask)
    
    # Verify realistic transformation
    total_diff = np.sum(cv2.absdiff(final_result, original_image))
    print(f"FINAL: Realistic transformation complete: {total_diff}")
    
    return final_result

# Direct replacement function for your existing code
def replace_with_new_garment(customer_analysis: Dict, garment_analysis: Dict,
                           product_info: Dict, original_image: np.ndarray,
                           garment_types: List[str]) -> np.ndarray:
    """
    DIRECT REPLACEMENT for your current virtual try-on function
    Use this to replace your existing process_virtual_tryon call
    """
    
    print("DEBUG: replace_with_new_garment CALLED")
    print(f"DEBUG: Product name: {product_info.get('name')}")
    print(f"DEBUG: Image shape: {original_image.shape}")
    
    try:
        # Use the realistic garment replacement system
        result = replace_garment_with_realism(
            customer_analysis=customer_analysis,
            garment_analysis=garment_analysis,
            product_info=product_info,
            original_image=original_image,
            garment_types=garment_types
        )
        
        print("DEBUG: Realistic replacement returned")
        print(f"DEBUG: Result shape: {result.shape}")
        
        # Check if result is different from input
        diff = np.sum(cv2.absdiff(result, original_image))
        print(f"DEBUG: Total visual difference: {diff}")
        
        return result
        
    except Exception as e:
        print(f"DEBUG: ERROR in realistic replacement: {str(e)}")
        import traceback
        traceback.print_exc()
        return original_image

# Legacy compatibility function
def process_complete_garment_replacement(customer_analysis: Dict, garment_analysis: Dict,
                                       product_info: Dict, original_image: np.ndarray,
                                       garment_types: List[str]) -> np.ndarray:
    """
    Legacy compatibility wrapper
    """
    print("DEBUG: process_complete_garment_replacement CALLED")
    return replace_with_new_garment(
        customer_analysis=customer_analysis,
        garment_analysis=garment_analysis,
        product_info=product_info,
        original_image=original_image,
        garment_types=garment_types
    )