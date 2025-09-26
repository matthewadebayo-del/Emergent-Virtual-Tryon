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
        COMPLETE garment replacement - removes original and adds new garment
        """
        
        try:
            print("DEBUG: replace_garment_completely METHOD CALLED")
            self.logger.info("🔥 DEBUG: PracticalGarmentReplacer.replace_garment_completely started")
            self.logger.info("[REPLACE] Starting COMPLETE garment replacement...")
            
            # STEP 1: Force correct color detection
            try:
                print("DEBUG: Starting color detection...")
                correct_color = self._force_correct_color(product_info, garment_analysis)
                self.logger.info(f"[REPLACE] Using color: {correct_color}")
                print(f"DEBUG: Color detection successful: {correct_color}")
            except Exception as e:
                print(f"DEBUG: Color detection failed: {e}")
                self.logger.error(f"[REPLACE] Color detection failed: {e}")
                return original_image
            
            # STEP 2: Create aggressive removal mask
            try:
                print("DEBUG: Starting mask creation...")
                print(f"DEBUG: Customer analysis keys: {list(customer_analysis.keys())}")
                pose_landmarks = customer_analysis.get('pose_landmarks', {})
                print(f"DEBUG: Pose landmarks type: {type(pose_landmarks)}")
                print(f"DEBUG: Pose landmarks sample: {dict(list(pose_landmarks.items())[:2]) if isinstance(pose_landmarks, dict) else 'Not a dict'}")
                
                removal_mask = self._create_complete_removal_mask(customer_analysis, original_image.shape)
                mask_area = np.sum(removal_mask > 128) if removal_mask is not None else 0
                print(f"DEBUG: Mask created - area: {mask_area} pixels")
                if mask_area == 0:
                    print("DEBUG: MASK IS EMPTY - returning original")
                    return original_image
            except Exception as e:
                print(f"DEBUG: Mask creation failed: {e}")
                self.logger.error(f"[REPLACE] Mask creation failed: {e}")
                return original_image
            
            # STEP 3: Remove original garment completely
            try:
                print("DEBUG: Starting garment removal...")
                body_without_garment = self._remove_original_garment_completely(original_image, removal_mask)
                removal_diff = np.sum(cv2.absdiff(body_without_garment, original_image))
                print(f"DEBUG: Garment removal diff: {removal_diff}")
            except Exception as e:
                print(f"DEBUG: Garment removal failed: {e}")
                self.logger.error(f"[REPLACE] Garment removal failed: {e}")
                return original_image
            
            # STEP 4: Create new garment with correct color
            try:
                print("DEBUG: Starting new garment creation...")
                new_garment = self._create_new_garment(correct_color, removal_mask, original_image.shape)
                garment_pixels = np.sum(new_garment > 0)
                print(f"DEBUG: New garment created - non-zero pixels: {garment_pixels}")
            except Exception as e:
                print(f"DEBUG: New garment creation failed: {e}")
                self.logger.error(f"[REPLACE] New garment creation failed: {e}")
                return original_image
            
            # STEP 5: Apply new garment with strong replacement
            try:
                print("DEBUG: Starting garment application...")
                result = self._apply_new_garment_strongly(
                    body_without_garment, new_garment, removal_mask
                )
                application_diff = np.sum(cv2.absdiff(result, body_without_garment))
                print(f"DEBUG: Garment application diff: {application_diff}")
            except Exception as e:
                print(f"DEBUG: Garment application failed: {e}")
                self.logger.error(f"[REPLACE] Garment application failed: {e}")
                return original_image
            
            # STEP 6: Final realism adjustments
            try:
                result = self._apply_final_realism_adjustments(result, original_image, removal_mask)
            except Exception as e:
                self.logger.error(f"[REPLACE] Final adjustments failed: {e}")
                return result  # Return partial result
            
            # Calculate and log the transformation
            try:
                total_change = np.sum(cv2.absdiff(result, original_image))
                self.logger.info(f"[REPLACE] Total visual change: {total_change}")
            except Exception as e:
                self.logger.error(f"[REPLACE] Change calculation failed: {e}")
            
            # Optional: Debug visual stages
            # result = self.debug_visual_stages(original_image, removal_mask, customer_analysis, garment_analysis, product_info)
            
            self.logger.info(f"[REPLACE] Complete replacement finished!")
            return result
            
        except Exception as e:
            self.logger.error(f"[REPLACE] Complete replacement failed: {e}")
            return original_image
    
    def _force_correct_color(self, product_info: Dict, garment_analysis: Dict) -> Tuple[int, int, int]:
        """
        FORCE correct color based on product name - no more gray for white shirts!
        """
        
        product_name = product_info.get('name', '').lower()
        self.logger.info(f"[COLOR] Product name: '{product_name}'")
        
        # ABSOLUTE priority to product name
        if any(word in product_name for word in ['white', 'blanc', 'blanco', 'wit']):
            color = (255, 255, 255)  # PURE WHITE
            self.logger.info(f"[COLOR] ✅ FORCING WHITE: {color}")
            return color
            
        elif any(word in product_name for word in ['black', 'noir', 'negro']):
            color = (10, 10, 10)     # PURE BLACK
            self.logger.info(f"[COLOR] ✅ FORCING BLACK: {color}")
            return color
            
        elif any(word in product_name for word in ['red', 'rouge', 'rojo']):
            color = (220, 20, 20)    # BRIGHT RED
            self.logger.info(f"[COLOR] ✅ FORCING RED: {color}")
            return color
            
        elif any(word in product_name for word in ['blue', 'bleu', 'azul']):
            color = (20, 20, 220)    # BRIGHT BLUE
            self.logger.info(f"[COLOR] ✅ FORCING BLUE: {color}")
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
                    self.logger.info(f"[COLOR] ✅ Light color ({r}, {g}, {b}) -> FORCING WHITE")
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
    
    def _apply_new_garment_strongly(self, body_image: np.ndarray, new_garment: np.ndarray, 
                                   mask: np.ndarray) -> np.ndarray:
        """
        Apply new garment with STRONG visual impact
        """
        
        # Create very strong application mask
        core_mask = (mask > 200).astype(np.float32)
        edge_mask = cv2.GaussianBlur(mask, (21, 21), 10).astype(np.float32) / 255.0
        
        # Strong application: 98% in core, 80% at edges
        final_mask = np.maximum(core_mask * 0.98, edge_mask * 0.8)
        
        # Apply new garment
        result = body_image.copy().astype(np.float32)
        garment_float = new_garment.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (
                body_image[:, :, c] * (1 - final_mask) +
                garment_float[:, :, c] * final_mask
            )
        
        # Verify strong application
        application_diff = np.sum(cv2.absdiff(result.astype(np.uint8), body_image))
        print(f"APPLICATION: Strong application difference: {application_diff}")
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
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
    
    def _create_new_garment(self, color: Tuple[int, int, int], mask: np.ndarray, 
                          image_shape: Tuple) -> np.ndarray:
        """
        Create new garment with the correct color and realistic details
        """
        
        height, width, _ = image_shape
        garment = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill garment area with the correct color
        garment_area = mask > 100
        garment[garment_area] = color
        
        # Add minimal texture for light colors, more for dark colors
        if color[0] > 200 and color[1] > 200 and color[2] > 200:  # Light/white
            texture_strength = 3  # Very minimal for white
        else:
            texture_strength = 8  # More texture for other colors
        
        # Add subtle fabric texture
        if texture_strength > 0:
            noise = np.random.normal(0, texture_strength, garment.shape)
            garment = np.clip(garment.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add basic garment features (seams, etc.)
        garment = self._add_realistic_garment_features(garment, mask, color)
        
        self.logger.info(f"[GARMENT] Created new garment with color {color}")
        return garment
    
    def _add_realistic_garment_features(self, garment: np.ndarray, mask: np.ndarray,
                                      base_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Add realistic garment features like seams, collar, etc.
        """
        
        # Create slightly darker color for seams/details
        seam_color = tuple(max(0, int(c * 0.9)) for c in base_color)
        
        # Find garment contours
        mask_binary = (mask > 200).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Significant garment area
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add shoulder seams (horizontal line across shoulders)
                shoulder_y = y + h // 6
                cv2.line(garment, (x + w//5, shoulder_y), (x + 4*w//5, shoulder_y), seam_color, 2)
                
                # Add side seams (vertical lines)
                left_seam_x = x + w // 5
                right_seam_x = x + 4*w // 5
                seam_start_y = y + h // 4
                seam_end_y = y + 3*h // 4
                
                cv2.line(garment, (left_seam_x, seam_start_y), (left_seam_x, seam_end_y), seam_color, 1)
                cv2.line(garment, (right_seam_x, seam_start_y), (right_seam_x, seam_end_y), seam_color, 1)
                
                # Add bottom hem
                hem_y = y + 4*h // 5
                cv2.line(garment, (x + w//5, hem_y), (x + 4*w//5, hem_y), seam_color, 2)
        
        return garment
    
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
        replacer = PracticalGarmentReplacer()
        print("DEBUG: PracticalGarmentReplacer instance created")
        
        result = replacer.replace_garment_completely(
            customer_analysis=customer_analysis,
            garment_analysis=garment_analysis,
            product_info=product_info,
            original_image=original_image,
            garment_types=garment_types
        )
        
        print("DEBUG: replace_garment_completely returned")
        print(f"DEBUG: Result shape: {result.shape}")
        
        # Check if result is different from input
        diff = np.sum(cv2.absdiff(result, original_image))
        print(f"DEBUG: Total visual difference: {diff}")
        
        return result
        
    except Exception as e:
        print(f"DEBUG: ERROR in complete replacement: {str(e)}")
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