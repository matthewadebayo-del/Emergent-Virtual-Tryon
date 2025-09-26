import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

class CompleteGarmentReplacement:
    """
    Complete garment replacement system that removes original clothing
    and replaces it with new garments for realistic virtual try-on
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def replace_garment_completely(self, customer_analysis: Dict, garment_analysis: Dict,
                                 product_info: Dict, original_image: np.ndarray,
                                 garment_types: List[str]) -> np.ndarray:
        """
        Complete garment replacement pipeline
        """
        self.logger.info("[REPLACEMENT] Starting complete garment replacement...")
        
        # Step 1: Extract original garment and create removal mask
        removal_mask = self._create_complete_removal_mask(
            customer_analysis, original_image.shape, garment_types
        )
        
        # Step 2: Remove original garment (inpaint the region)
        garment_removed_image = self._remove_original_garment(
            original_image, removal_mask
        )
        
        # Step 3: Generate new garment with correct properties
        new_garment = self._generate_realistic_garment(
            garment_analysis, product_info, original_image.shape, removal_mask
        )
        
        # Step 4: Apply lighting and body curvature to new garment
        fitted_garment = self._apply_body_fitting(
            new_garment, removal_mask, customer_analysis, original_image
        )
        
        # Step 5: Composite the new garment onto the body
        final_result = self._composite_new_garment(
            garment_removed_image, fitted_garment, removal_mask
        )
        
        # Step 6: Final touch-ups and realism
        final_result = self._apply_final_realism(
            final_result, original_image, removal_mask
        )
        
        self.logger.info("[REPLACEMENT] Complete garment replacement finished")
        return final_result
    
    def _create_complete_removal_mask(self, customer_analysis: Dict, 
                                    image_shape: Tuple, garment_types: List[str]) -> np.ndarray:
        """Create mask that covers the ENTIRE garment area for complete removal"""
        
        height, width, _ = image_shape
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        
        # Get landmark positions
        landmarks = {}
        for lm_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                       'left_elbow', 'right_elbow']:
            if lm_name in pose_landmarks:
                lm = pose_landmarks[lm_name]
                x = int(lm[0] * width) if isinstance(lm, (list, tuple)) else int(lm.get('x', 0) * width)
                y = int(lm[1] * height) if isinstance(lm, (list, tuple)) else int(lm.get('y', 0) * height)
                landmarks[lm_name] = (x, y)
        
        if len(landmarks) < 4:
            self.logger.error("[MASK] Insufficient landmarks for removal mask")
            return np.zeros((height, width), dtype=np.uint8)
        
        # Create AGGRESSIVE removal mask that covers entire torso
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate dimensions
        shoulder_width = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        torso_height = abs(landmarks['left_hip'][1] - landmarks['left_shoulder'][1])
        
        # MUCH more aggressive expansion for complete coverage
        horizontal_expansion = max(50, int(shoulder_width * 0.4))  # 40% wider than shoulders
        vertical_expansion = max(30, int(torso_height * 0.2))      # 20% taller
        
        # Create polygon that completely covers the garment area
        left_x = landmarks['left_shoulder'][0] - horizontal_expansion
        right_x = landmarks['right_shoulder'][0] + horizontal_expansion
        top_y = landmarks['left_shoulder'][1] - vertical_expansion
        bottom_y = max(landmarks['left_hip'][1], landmarks['right_hip'][1]) + vertical_expansion
        
        # Extend to sides based on elbow positions if available
        if 'left_elbow' in landmarks and 'right_elbow' in landmarks:
            left_x = min(left_x, landmarks['left_elbow'][0] - 20)
            right_x = max(right_x, landmarks['right_elbow'][0] + 20)
        
        # Create comprehensive coverage polygon
        polygon_points = np.array([
            (left_x, top_y),                    # Top-left
            (right_x, top_y),                   # Top-right
            (right_x + 20, (top_y + bottom_y) // 2),  # Right side expansion
            (right_x, bottom_y),                # Bottom-right
            (left_x, bottom_y),                 # Bottom-left
            (left_x - 20, (top_y + bottom_y) // 2),   # Left side expansion
        ], dtype=np.int32)
        
        # Ensure within image bounds
        polygon_points[:, 0] = np.clip(polygon_points[:, 0], 0, width - 1)
        polygon_points[:, 1] = np.clip(polygon_points[:, 1], 0, height - 1)
        
        # Fill the polygon
        cv2.fillPoly(mask, [polygon_points], 255)
        
        # Apply heavy smoothing for natural edges
        mask = cv2.GaussianBlur(mask, (41, 41), 20)
        
        # Log mask statistics
        mask_area = np.sum(mask > 128)
        coverage = (mask_area / (width * height)) * 100
        self.logger.info(f"[MASK] Removal mask: {mask_area} pixels ({coverage:.1f}% coverage)")
        
        return mask
    
    def _remove_original_garment(self, original_image: np.ndarray, 
                               removal_mask: np.ndarray) -> np.ndarray:
        """Remove the original garment using inpainting and skin tone estimation"""
        
        # Create inpainting mask (areas to fill)
        inpaint_mask = (removal_mask > 200).astype(np.uint8) * 255
        
        # Estimate skin tone from visible areas (face, arms if visible)
        skin_tone = self._estimate_skin_tone(original_image, removal_mask)
        
        # Use OpenCV inpainting to fill the garment area
        inpainted = cv2.inpaint(original_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
        
        # Enhance with skin tone in the torso area
        torso_area = removal_mask > 150
        inpainted[torso_area] = (
            inpainted[torso_area].astype(np.float32) * 0.7 + 
            np.array(skin_tone, dtype=np.float32) * 0.3
        ).astype(np.uint8)
        
        self.logger.info(f"[REMOVAL] Original garment removed, skin tone: {skin_tone}")
        return inpainted
    
    def _estimate_skin_tone(self, image: np.ndarray, garment_mask: np.ndarray) -> Tuple[int, int, int]:
        """Estimate skin tone from non-garment areas"""
        
        # Create mask for potential skin areas (exclude garment region)
        height, width = image.shape[:2]
        skin_mask = np.ones((height, width), dtype=np.uint8) * 255
        skin_mask[garment_mask > 100] = 0  # Exclude garment area
        
        # Focus on upper portion (likely face/neck area)
        skin_mask[height//2:] = 0  # Only upper half
        
        # Get skin pixels
        skin_pixels = image[skin_mask > 0]
        
        if len(skin_pixels) > 0:
            # Calculate median skin tone (more robust than mean)
            skin_tone = tuple(int(x) for x in np.median(skin_pixels, axis=0))
        else:
            # Fallback to generic skin tone
            skin_tone = (210, 180, 160)  # Light skin tone fallback
        
        self.logger.info(f"[SKIN] Estimated skin tone: {skin_tone}")
        return skin_tone
    
    def _generate_realistic_garment(self, garment_analysis: Dict, product_info: Dict,
                                  image_shape: Tuple, mask: np.ndarray) -> np.ndarray:
        """Generate realistic garment with CORRECT colors"""
        
        height, width, _ = image_shape
        garment = np.zeros((height, width, 3), dtype=np.uint8)
        
        # FIXED COLOR LOGIC - Product name takes absolute priority
        product_name = product_info.get('name', '').lower()
        self.logger.info(f"[GARMENT] Processing product: '{product_name}'")
        
        # 1. STRICT product name matching (highest priority)
        if any(color_word in product_name for color_word in ['white', 'blanc', 'blanco']):
            base_color = (255, 255, 255)  # Pure white
            self.logger.info("[GARMENT] ✅ FORCING WHITE from product name")
        elif any(color_word in product_name for color_word in ['black', 'noir', 'negro']):
            base_color = (20, 20, 20)     # Near black
            self.logger.info("[GARMENT] ✅ FORCING BLACK from product name")
        elif any(color_word in product_name for color_word in ['red', 'rouge', 'rojo']):
            base_color = (200, 40, 40)    # Red
            self.logger.info("[GARMENT] ✅ FORCING RED from product name")
        elif any(color_word in product_name for color_word in ['blue', 'bleu', 'azul']):
            base_color = (40, 40, 200)    # Blue
            self.logger.info("[GARMENT] ✅ FORCING BLUE from product name")
        else:
            # 2. Fallback to analysis only if no color in name
            dominant_colors = garment_analysis.get('dominant_colors', [])
            if dominant_colors:
                base_color = dominant_colors[0]
                self.logger.info(f"[GARMENT] Using analyzed color: {base_color}")
            else:
                base_color = (128, 128, 128)  # Gray fallback
                self.logger.warning("[GARMENT] Using gray fallback color")
        
        # Fill the garment area with base color
        garment_area = mask > 100
        garment[garment_area] = base_color
        
        # Add fabric texture (minimal for white to maintain clarity)
        texture_strength = 0.02 if base_color[0] > 200 and base_color[1] > 200 and base_color[2] > 200 else 0.05
        if texture_strength > 0:
            noise = np.random.normal(0, texture_strength * 255, garment.shape)
            garment = np.clip(garment.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add basic garment details (seams, etc.)
        garment = self._add_garment_details(garment, mask, base_color)
        
        self.logger.info(f"[GARMENT] Generated garment with color: {base_color}")
        return garment
    
    def _add_garment_details(self, garment: np.ndarray, mask: np.ndarray, 
                           base_color: Tuple[int, int, int]) -> np.ndarray:
        """Add realistic garment details like seams, hems, etc."""
        
        # Create slightly darker color for seams
        seam_color = tuple(max(0, int(c * 0.92)) for c in base_color)
        
        # Find garment contours
        mask_binary = (mask > 150).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Significant garment area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add shoulder seams (horizontal lines at top)
                shoulder_y = y + h // 8
                cv2.line(garment, (x + w//4, shoulder_y), (x + 3*w//4, shoulder_y), seam_color, 2)
                
                # Add side seams (vertical lines)
                left_seam_x = x + w // 6
                right_seam_x = x + 5*w // 6
                cv2.line(garment, (left_seam_x, y + h//4), (left_seam_x, y + 3*h//4), seam_color, 1)
                cv2.line(garment, (right_seam_x, y + h//4), (right_seam_x, y + 3*h//4), seam_color, 1)
                
                # Add bottom hem
                hem_y = y + 7*h // 8
                cv2.line(garment, (x + w//6, hem_y), (x + 5*w//6, hem_y), seam_color, 2)
        
        return garment
    
    def _apply_body_fitting(self, garment: np.ndarray, mask: np.ndarray,
                          customer_analysis: Dict, original_image: np.ndarray) -> np.ndarray:
        """Apply body curvature and lighting to make garment look fitted"""
        
        # Extract lighting from original image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        lighting_map = cv2.GaussianBlur(original_gray, (51, 51), 25).astype(np.float32) / 255.0
        
        # Apply lighting to garment
        fitted_garment = garment.copy().astype(np.float32)
        for c in range(3):
            fitted_garment[:, :, c] = fitted_garment[:, :, c] * lighting_map
        
        # Add body curvature shadows
        fitted_garment = self._add_body_shadows(fitted_garment, mask, customer_analysis)
        
        return np.clip(fitted_garment, 0, 255).astype(np.uint8)
    
    def _add_body_shadows(self, garment: np.ndarray, mask: np.ndarray,
                        customer_analysis: Dict) -> np.ndarray:
        """Add realistic body shadows and curvature"""
        
        pose_landmarks = customer_analysis.get('pose_landmarks', {})
        
        if 'left_shoulder' in pose_landmarks and 'right_shoulder' in pose_landmarks:
            height, width = garment.shape[:2]
            
            # Create shadow gradients from center outward
            center_x = width // 2
            center_y = height // 2
            
            # Create radial shadow
            y, x = np.ogrid[:height, :width]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Normalize distances and create shadow effect
            shadow_strength = 1.0 - (distances / max_distance) * 0.3  # Max 30% darkening
            shadow_strength = np.clip(shadow_strength, 0.7, 1.0)
            
            # Apply shadow only to garment area
            garment_area = mask > 100
            for c in range(3):
                garment[:, :, c][garment_area] = (garment[:, :, c][garment_area] * shadow_strength[garment_area]).astype(np.float32)
        
        return garment
    
    def _composite_new_garment(self, body_image: np.ndarray, fitted_garment: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """Composite the new garment onto the body"""
        
        # Create smooth blending mask
        blend_mask = cv2.GaussianBlur(mask, (31, 31), 15).astype(np.float32) / 255.0
        
        # Strong compositing in the core area
        core_mask = (mask > 200).astype(np.float32)
        final_mask = np.maximum(blend_mask, core_mask * 0.95)  # At least 95% in core
        
        # Composite
        result = body_image.copy().astype(np.float32)
        garment_float = fitted_garment.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (
                body_image[:, :, c] * (1 - final_mask) +
                garment_float[:, :, c] * final_mask
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_final_realism(self, result: np.ndarray, original_image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """Apply final touches for maximum realism"""
        
        # Color temperature matching
        result = self._match_color_temperature(result, original_image, mask)
        
        # Edge enhancement
        result = self._enhance_edges(result, mask)
        
        # Final sharpening
        result = self._apply_subtle_sharpening(result)
        
        return result
    
    def _match_color_temperature(self, result: np.ndarray, original: np.ndarray,
                               mask: np.ndarray) -> np.ndarray:
        """Match color temperature of the scene"""
        
        # Get average color temperature from non-garment areas
        non_garment_area = mask < 50
        if np.sum(non_garment_area) > 0:
            original_avg = np.mean(original[non_garment_area], axis=0)
            result_avg = np.mean(result[non_garment_area], axis=0)
            
            # Calculate color shift
            color_shift = original_avg / (result_avg + 1e-6)
            
            # Apply subtle color temperature correction
            garment_area = mask > 100
            for c in range(3):
                result[:, :, c][garment_area] = (result[:, :, c][garment_area] * (color_shift[c] * 0.1 + 0.9)).astype(np.uint8)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _enhance_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhance garment edges for realism"""
        
        # Find mask edges
        edges = cv2.Canny((mask > 150).astype(np.uint8), 50, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        
        # Apply subtle edge enhancement
        enhanced = image.copy()
        edge_pixels = edges > 0
        enhanced[edge_pixels] = np.clip(enhanced[edge_pixels] * 1.05, 0, 255)  # 5% brighter
        
        return enhanced
    
    def _apply_subtle_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle sharpening for final result"""
        
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)
        
        # Blend original and sharpened (subtle effect)
        return cv2.addWeighted(image, 0.8, sharpened, 0.2, 0)


# Integration function for your existing system
def process_complete_garment_replacement(customer_analysis: Dict, garment_analysis: Dict,
                                       product_info: Dict, original_image: np.ndarray,
                                       garment_types: List[str]) -> np.ndarray:
    """
    Complete garment replacement function to replace your current system
    This removes the original garment completely and replaces it with the new one
    """
    
    processor = CompleteGarmentReplacement()
    return processor.replace_garment_completely(
        customer_analysis=customer_analysis,
        garment_analysis=garment_analysis,
        product_info=product_info,
        original_image=original_image,
        garment_types=garment_types
    )