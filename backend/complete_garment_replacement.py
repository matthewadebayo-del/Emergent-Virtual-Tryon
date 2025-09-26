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
            print("🔥 DEBUG: replace_garment_completely METHOD CALLED")
            self.logger.info("🔥 DEBUG: PracticalGarmentReplacer.replace_garment_completely started")
            self.logger.info("[REPLACE] Starting COMPLETE garment replacement...")
            
            # STEP 1: Force correct color detection
            try:
                print("🔥 DEBUG: Starting color detection...")
                correct_color = self._force_correct_color(product_info, garment_analysis)
                self.logger.info(f"[REPLACE] Using color: {correct_color}")
                print(f"🔥 DEBUG: Color detection successful: {correct_color}")
            except Exception as e:
                print(f"🔥 DEBUG: Color detection failed: {e}")
                self.logger.error(f"[REPLACE] Color detection failed: {e}")
                return original_image
            
            # STEP 2: Create aggressive removal mask
            try:
                removal_mask = self._create_complete_removal_mask(customer_analysis, original_image.shape)
            except Exception as e:
                self.logger.error(f"[REPLACE] Mask creation failed: {e}")
                return original_image
            
            # STEP 3: Remove original garment completely
            try:
                body_without_garment = self._remove_original_garment_completely(original_image, removal_mask)
            except Exception as e:
                self.logger.error(f"[REPLACE] Garment removal failed: {e}")
                return original_image
            
            # STEP 4: Create new garment with correct color
            try:
                new_garment = self._create_new_garment(correct_color, removal_mask, original_image.shape)
            except Exception as e:
                self.logger.error(f"[REPLACE] New garment creation failed: {e}")
                return original_image
            
            # STEP 5: Apply new garment with strong replacement
            try:
                result = self._apply_garment_with_strong_replacement(
                    body_without_garment, new_garment, removal_mask, original_image
                )
            except Exception as e:
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
        Create AGGRESSIVE mask that covers the ENTIRE garment area for complete removal
        """
        
        height, width, _ = image_shape
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        
        # Extract landmark coordinates
        landmarks = {}
        for lm_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if lm_name in pose_landmarks:
                lm = pose_landmarks[lm_name]
                x = int(lm[0] * width) if isinstance(lm, (list, tuple)) else int(lm.get('x', 0) * width)
                y = int(lm[1] * height) if isinstance(lm, (list, tuple)) else int(lm.get('y', 0) * height)
                landmarks[lm_name] = (x, y)
                self.logger.info(f"[MASK] {lm_name}: ({x}, {y})")
        
        if len(landmarks) < 4:
            self.logger.error(f"[MASK] Insufficient landmarks: {list(landmarks.keys())}")
            # Create fallback mask covering center area
            return self._create_fallback_mask(height, width)
        
        # Calculate dimensions
        shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        torso_height = abs(landmarks['left_hip'][1] - landmarks['left_shoulder'][1])
        
        # VERY AGGRESSIVE expansion for complete coverage
        horizontal_expansion = max(80, int(shoulder_width * 0.6))  # 60% wider than shoulders
        vertical_expansion = max(50, int(torso_height * 0.3))      # 30% taller
        
        self.logger.info(f"[MASK] Shoulder width: {shoulder_width}px, expansion: {horizontal_expansion}px")
        
        # Create comprehensive polygon
        ls = landmarks['left_shoulder']
        rs = landmarks['right_shoulder']
        lh = landmarks['left_hip']
        rh = landmarks['right_hip']
        
        # Expanded polygon points
        polygon_points = np.array([
            # Top edge (expanded up and out)
            (ls[0] - horizontal_expansion, ls[1] - vertical_expansion),
            (rs[0] + horizontal_expansion, rs[1] - vertical_expansion),
            
            # Right side (expanded out)
            (rs[0] + horizontal_expansion + 30, (rs[1] + rh[1]) // 2),
            
            # Bottom edge (expanded down and out)
            (rh[0] + horizontal_expansion//2, rh[1] + vertical_expansion),
            (lh[0] - horizontal_expansion//2, lh[1] + vertical_expansion),
            
            # Left side (expanded out)
            (ls[0] - horizontal_expansion - 30, (ls[1] + lh[1]) // 2),
        ], dtype=np.int32)
        
        # Ensure within image bounds
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, width - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, height - 1)
        
        # Create and fill mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        
        # Heavy smoothing for natural edges
        mask = cv2.GaussianBlur(mask, (51, 51), 25)
        
        # Log mask statistics
        mask_area = np.sum(mask > 128)
        coverage = (mask_area / (width * height)) * 100
        self.logger.info(f"[MASK] Removal mask: {mask_area} pixels ({coverage:.1f}% coverage)")
        
        # Ensure mask is large enough
        if coverage < 15:  # Less than 15% is too small
            self.logger.warning(f"[MASK] Mask too small, expanding further...")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
            mask = cv2.dilate(mask, kernel, iterations=3)
            final_coverage = (np.sum(mask > 128) / (width * height)) * 100
            self.logger.info(f"[MASK] Expanded to {final_coverage:.1f}% coverage")
        
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
        Completely remove the original garment using inpainting and skin tone estimation
        """
        
        # Create inpainting mask (binary mask for cv2.inpaint)
        inpaint_mask = (removal_mask > 200).astype(np.uint8) * 255
        
        # Estimate skin tone from visible areas
        skin_tone = self._estimate_skin_tone(original_image, removal_mask)
        
        # Method 1: Use OpenCV inpainting
        inpainted = cv2.inpaint(original_image, inpaint_mask, 5, cv2.INPAINT_TELEA)
        
        # Method 2: Fill with estimated skin tone
        skin_filled = original_image.copy()
        garment_area = removal_mask > 150
        skin_filled[garment_area] = skin_tone
        
        # Blend inpainted and skin-filled results
        blend_ratio = 0.3  # 30% inpainted, 70% skin tone
        result = cv2.addWeighted(inpainted, blend_ratio, skin_filled, 1 - blend_ratio, 0)
        
        self.logger.info(f"[REMOVAL] Original garment removed, using skin tone: {skin_tone}")
        return result
    
    def _estimate_skin_tone(self, image: np.ndarray, garment_mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Estimate skin tone from non-garment areas (face, arms, etc.)
        """
        
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
            # Use median for robustness against outliers
            skin_tone = tuple(int(np.median(skin_pixels, axis=0)))
        else:
            # Fallback to a neutral skin tone
            skin_tone = (200, 170, 150)
        
        self.logger.info(f"[SKIN] Estimated skin tone: {skin_tone} from {len(skin_pixels)} pixels")
        return skin_tone
    
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
    
    print("🔥 DEBUG: replace_with_new_garment CALLED")
    print(f"🔥 DEBUG: Product name: {product_info.get('name')}")
    print(f"🔥 DEBUG: Image shape: {original_image.shape}")
    
    try:
        replacer = PracticalGarmentReplacer()
        print("🔥 DEBUG: PracticalGarmentReplacer instance created")
        
        result = replacer.replace_garment_completely(
            customer_analysis=customer_analysis,
            garment_analysis=garment_analysis,
            product_info=product_info,
            original_image=original_image,
            garment_types=garment_types
        )
        
        print("🔥 DEBUG: replace_garment_completely returned")
        print(f"🔥 DEBUG: Result shape: {result.shape}")
        
        # Check if result is different from input
        diff = np.sum(cv2.absdiff(result, original_image))
        print(f"🔥 DEBUG: Total visual difference: {diff}")
        
        return result
        
    except Exception as e:
        print(f"🔥 DEBUG: ERROR in complete replacement: {str(e)}")
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
    print("🔥 DEBUG: process_complete_garment_replacement CALLED")
    return replace_with_new_garment(
        customer_analysis=customer_analysis,
        garment_analysis=garment_analysis,
        product_info=product_info,
        original_image=original_image,
        garment_types=garment_types
    )