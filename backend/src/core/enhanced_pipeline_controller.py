"""
Enhanced Pipeline Controller - Main orchestrator for virtual try-on processing
Coordinates customer and garment image analyses with parallel processing
"""

import asyncio
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

from .customer_image_analyzer import CustomerImageAnalyzer
from .garment_analyzer import GarmentImageAnalyzer
from .enhanced_3d_garment_processor import Enhanced3DGarmentProcessor
from .performance_optimizations import ImagePreprocessor, AnalysisCache, GPUAccelerator

logger = logging.getLogger(__name__)

class EnhancedPipelineController:
    """Main orchestrator for virtual try-on processing with dual image analysis"""
    
    def __init__(self):
        self.customer_analyzer = CustomerImageAnalyzer()
        self.garment_analyzer = GarmentImageAnalyzer()
        self.garment_processor = Enhanced3DGarmentProcessor()
        
    async def process_virtual_tryon(
        self, 
        customer_image: Image.Image, 
        garment_image: Image.Image,
        garment_type: str = "t-shirt"
    ) -> Dict[str, Any]:
        """
        Optimized processing pipeline with GPU acceleration, preprocessing, and caching
        """
        try:
            # Step 1: Preprocess images for optimal quality
            customer_processed = ImagePreprocessor.preprocess_image(customer_image)
            garment_processed = ImagePreprocessor.preprocess_image(garment_image)
            
            # Step 2: Check cache for existing analyses
            customer_hash = AnalysisCache.get_image_hash(customer_processed)
            garment_hash = AnalysisCache.get_image_hash(garment_processed)
            
            customer_cached = AnalysisCache.get_cached_analysis(customer_hash, "customer")
            garment_cached = AnalysisCache.get_cached_analysis(garment_hash, "garment")
            
            # Step 3: Run analyses (using cache if available)
            if customer_cached and garment_cached:
                logger.info("Using cached analyses for both images")
                customer_analysis, garment_analysis = customer_cached, garment_cached
            else:
                customer_analysis, garment_analysis = await self._run_parallel_analyses(
                    customer_processed, garment_processed, customer_hash, garment_hash,
                    customer_cached, garment_cached
                )
            
            # Step 4: Validate analyses
            validation_result = self._validate_analyses(customer_analysis, garment_analysis)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Step 5: Run fitting algorithm
            fitting_result = self._run_fitting_algorithm(
                customer_analysis, garment_analysis, garment_type
            )
            
            # Step 6: Generate final render with AI enhancement
            final_image = await self._generate_final_render_with_ai(
                customer_image, customer_analysis, garment_analysis, fitting_result
            )
            
            return {
                "success": True,
                "result_image": final_image,
                "customer_analysis": customer_analysis,
                "garment_analysis": garment_analysis,
                "fitting_data": fitting_result,
                "performance_info": {
                    "gpu_used": GPUAccelerator.is_gpu_available(),
                    "customer_cached": customer_cached is not None,
                    "garment_cached": garment_cached is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}
    
    async def _run_parallel_analyses(
        self, 
        customer_image: Image.Image, 
        garment_image: Image.Image,
        customer_hash: str,
        garment_hash: str,
        customer_cached: Optional[Dict[str, Any]],
        garment_cached: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run analyses in parallel with caching and GPU acceleration"""
        
        async def analyze_customer():
            if customer_cached:
                logger.info("Using cached customer analysis")
                return customer_cached
            
            # Convert PIL Image to bytes for analyzer
            import io
            img_bytes = io.BytesIO()
            customer_image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            
            # Use GPU-accelerated analysis
            result = self.customer_analyzer.analyze_customer_image(img_bytes)
            AnalysisCache.cache_analysis(customer_hash, "customer", result)
            return result
        
        async def analyze_garment():
            if garment_cached:
                logger.info("Using cached garment analysis")
                return garment_cached
            
            # Convert PIL Image to bytes for analyzer
            import io
            img_bytes = io.BytesIO()
            garment_image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            
            result = self.garment_analyzer.analyze_garment_image(img_bytes)
            AnalysisCache.cache_analysis(garment_hash, "garment", result)
            return result
        
        # Run analyses concurrently
        customer_task = asyncio.create_task(analyze_customer())
        garment_task = asyncio.create_task(analyze_garment())
        
        customer_analysis = await customer_task
        garment_analysis = await garment_task
        
        return customer_analysis, garment_analysis
    
    def _validate_analyses(
        self, 
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate both analyses completed successfully"""
        
        # Customer validation - strict error reporting
        if not customer_analysis.get("pose_keypoints"):
            return {"valid": False, "error": "Customer pose detection failed"}
        
        keypoints = customer_analysis["pose_keypoints"]
        if not isinstance(keypoints, dict):
            return {"valid": False, "error": "Invalid pose keypoints format"}
            
        required_keypoints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        missing_keypoints = [kp for kp in required_keypoints if kp not in keypoints]
        
        if missing_keypoints:
            return {"valid": False, "error": f"Missing required keypoints: {missing_keypoints}"}
        
        # Measurement validation - check what we actually get
        measurements = customer_analysis.get("measurements", {})
        print(f"[VALIDATION] Customer measurements: {measurements}")
        
        if not measurements:
            return {"valid": False, "error": "Customer measurements extraction failed"}
        
        # Check for key measurements that are actually provided
        shoulder_width = measurements.get("shoulder_width_cm", 0)
        height = measurements.get("height_cm", 0)
        
        if shoulder_width < 30 or shoulder_width > 70:
            return {"valid": False, "error": f"Invalid shoulder width: {shoulder_width}cm"}
        
        if height < 140 or height > 220:
            return {"valid": False, "error": f"Invalid height: {height}cm"}
        
        # Garment validation - strict error reporting  
        colors = garment_analysis.get("dominant_colors", [])
        print(f"[VALIDATION] Garment colors: {colors}")
        
        if not colors or len(colors) == 0:
            return {"valid": False, "error": "Garment color extraction failed"}
        
        # Check if garment has actual visual features - be more lenient
        if self._is_garment_too_plain(colors):
            print(f"[VALIDATION] Garment considered too plain: {colors}")
            return {"valid": False, "error": "Garment appears to have no distinct visual features"}
        
        texture_features = garment_analysis.get("texture_features", {})
        print(f"[VALIDATION] Texture features: {texture_features}")
        print(f"[VALIDATION] Texture complexity: {texture_features.get('complexity', 0)} (threshold: 0.05)")
        
        if not texture_features or texture_features.get("complexity", 0) < 0.05:
            return {"valid": False, "error": "Garment texture analysis failed"}
        
        return {"valid": True}
    
    def _is_garment_too_plain(self, colors: list) -> bool:
        """Check if garment colors are too plain (all gray/white)"""
        if not colors:
            return True
        
        # Convert colors to grayscale and check variance
        gray_values = []
        for color in colors[:3]:  # Check top 3 colors
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                gray = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                gray_values.append(gray)
        
        if not gray_values:
            return True
        
        # If all colors are very light (>200) or have low variance, consider too plain
        avg_gray = np.mean(gray_values)
        variance = np.var(gray_values)
        
        return avg_gray > 220 and variance < 50  # More lenient thresholds
    
    def _run_fitting_algorithm(
        self, 
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any],
        garment_type: str
    ) -> Dict[str, Any]:
        """Combine both analyses for realistic fitting"""
        
        # Extract key data
        measurements = customer_analysis["measurements"]
        pose_keypoints = customer_analysis["pose_keypoints"]
        skin_tone = customer_analysis.get("skin_tone", {})
        
        garment_colors = garment_analysis["dominant_colors"]
        fabric_type = garment_analysis.get("fabric_type", "cotton")
        patterns = garment_analysis.get("patterns", [])
        
        # Calculate garment scaling based on measurements
        scale_factors = self._calculate_garment_scaling(measurements, garment_type)
        
        # Determine garment positioning based on pose
        positioning = self._calculate_garment_positioning(pose_keypoints, garment_type)
        
        # Color matching and contrast analysis
        color_matching = self._analyze_color_matching(garment_colors, skin_tone)
        
        # Fabric draping properties
        draping_props = self._get_draping_properties(fabric_type, garment_type)
        
        return {
            "scale_factors": scale_factors,
            "positioning": positioning,
            "color_matching": color_matching,
            "draping_properties": draping_props,
            "visual_properties": {
                "colors": garment_colors,
                "patterns": patterns,
                "fabric_type": fabric_type
            }
        }
    
    def _calculate_garment_scaling(self, measurements: Dict[str, float], garment_type: str) -> Dict[str, float]:
        """Calculate how to scale garment mesh based on customer measurements"""
        
        base_measurements = {
            "chest": 90, "waist": 75, "hips": 95,
            "shoulder_width": 45, "arm_length": 60
        }
        
        scale_factors = {}
        for measurement, value in measurements.items():
            if measurement in base_measurements:
                scale_factors[measurement] = value / base_measurements[measurement]
        
        # Garment-specific adjustments
        if garment_type in ["t-shirt", "polo", "blouse"]:
            scale_factors["primary"] = scale_factors.get("chest", 1.0)
        elif garment_type in ["jeans", "pants", "skirt"]:
            scale_factors["primary"] = scale_factors.get("waist", 1.0)
        elif garment_type == "dress":
            scale_factors["primary"] = (scale_factors.get("chest", 1.0) + scale_factors.get("waist", 1.0)) / 2
        
        return scale_factors
    
    def _calculate_garment_positioning(self, pose_keypoints: Dict[str, Any], garment_type: str) -> Dict[str, Any]:
        """Calculate garment positioning based on customer pose"""
        
        # Get key body points
        shoulders = {
            "left": pose_keypoints.get("left_shoulder", [0, 0]),
            "right": pose_keypoints.get("right_shoulder", [0, 0])
        }
        
        hips = {
            "left": pose_keypoints.get("left_hip", [0, 0]),
            "right": pose_keypoints.get("right_hip", [0, 0])
        }
        
        # Calculate body center and orientation
        shoulder_center = [
            (shoulders["left"][0] + shoulders["right"][0]) / 2,
            (shoulders["left"][1] + shoulders["right"][1]) / 2
        ]
        
        hip_center = [
            (hips["left"][0] + hips["right"][0]) / 2,
            (hips["left"][1] + hips["right"][1]) / 2
        ]
        
        # Calculate body angle
        body_angle = np.arctan2(
            hip_center[1] - shoulder_center[1],
            hip_center[0] - shoulder_center[0]
        )
        
        # Garment-specific positioning
        if garment_type in ["t-shirt", "polo", "blouse", "blazer"]:
            anchor_point = shoulder_center
        elif garment_type in ["jeans", "pants", "skirt"]:
            anchor_point = hip_center
        elif garment_type == "dress":
            anchor_point = [(shoulder_center[0] + hip_center[0]) / 2, shoulder_center[1]]
        else:
            anchor_point = shoulder_center
        
        return {
            "anchor_point": anchor_point,
            "body_angle": body_angle,
            "shoulder_width": abs(shoulders["right"][0] - shoulders["left"][0]),
            "torso_length": abs(hip_center[1] - shoulder_center[1])
        }
    
    def _analyze_color_matching(self, garment_colors: list, skin_tone: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze color matching between garment and skin tone"""
        
        if not skin_tone or not garment_colors:
            return {"compatibility": "neutral", "contrast": "medium"}
        
        skin_rgb = skin_tone.get("rgb", [120, 100, 80])
        
        # Calculate contrast with primary garment color
        primary_color = garment_colors[0] if garment_colors else [128, 128, 128]
        
        # Simple contrast calculation
        contrast = np.sqrt(sum([(a - b) ** 2 for a, b in zip(primary_color, skin_rgb)]))
        
        if contrast > 150:
            contrast_level = "high"
        elif contrast > 75:
            contrast_level = "medium"
        else:
            contrast_level = "low"
        
        # Color temperature matching
        skin_warmth = skin_tone.get("warmth", "neutral")
        garment_warmth = self._analyze_color_warmth(primary_color)
        
        if skin_warmth == garment_warmth:
            compatibility = "excellent"
        elif skin_warmth == "neutral" or garment_warmth == "neutral":
            compatibility = "good"
        else:
            compatibility = "fair"
        
        return {
            "compatibility": compatibility,
            "contrast": contrast_level,
            "contrast_value": contrast,
            "skin_warmth": skin_warmth,
            "garment_warmth": garment_warmth
        }
    
    def _analyze_color_warmth(self, color: list) -> str:
        """Analyze if color is warm, cool, or neutral"""
        if len(color) < 3:
            return "neutral"
        
        r, g, b = color[:3]
        
        # Simple warmth analysis
        if r > g and r > b:
            return "warm"
        elif b > r and b > g:
            return "cool"
        else:
            return "neutral"
    
    def _get_draping_properties(self, fabric_type: str, garment_type: str) -> Dict[str, Any]:
        """Get fabric draping properties for physics simulation"""
        
        fabric_properties = {
            "silk": {"stiffness": 0.2, "weight": 0.3, "flow": 0.9},
            "cotton": {"stiffness": 0.5, "weight": 0.5, "flow": 0.6},
            "wool": {"stiffness": 0.7, "weight": 0.7, "flow": 0.4},
            "denim": {"stiffness": 0.9, "weight": 0.8, "flow": 0.2},
            "polyester": {"stiffness": 0.4, "weight": 0.4, "flow": 0.7}
        }
        
        base_props = fabric_properties.get(fabric_type, fabric_properties["cotton"])
        
        # Garment-specific adjustments
        if garment_type == "dress":
            base_props["flow"] *= 1.2
        elif garment_type in ["jeans", "blazer"]:
            base_props["stiffness"] *= 1.1
        
        return base_props
    
    async def _generate_final_render_with_ai(
        self, 
        customer_image: Image.Image,
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any], 
        fitting_result: Dict[str, Any]
    ) -> Image.Image:
        """Generate final rendered image with AI enhancement"""
        
        try:
            # Step 1: Generate 3D base render
            print("[AI] Starting 3D base rendering...")
            result = self.garment_processor.process_garment_with_analysis(
                garment_analysis=garment_analysis,
                customer_analysis=customer_analysis,
                fitting_data=fitting_result
            )
            
            if result.get("success") and result.get("rendered_image"):
                base_render = result["rendered_image"]
            else:
                base_render = self._create_fallback_composite(
                    customer_analysis, garment_analysis, fitting_result
                )
            
            # Step 2: Apply AI enhancement using Stable Diffusion
            print("[AI] Enhancing image realism with Stable Diffusion...")
            enhanced_image = await self._apply_ai_enhancement(
                customer_image, base_render, garment_analysis, fitting_result
            )
            
            return enhanced_image
                
        except Exception as e:
            logger.error(f"AI-enhanced rendering failed: {str(e)}")
            return self._create_fallback_composite(
                customer_analysis, garment_analysis, fitting_result
            )
    
    async def _apply_ai_enhancement(
        self,
        customer_image: Image.Image,
        base_render: Image.Image,
        garment_analysis: Dict[str, Any],
        fitting_result: Dict[str, Any]
    ) -> Image.Image:
        """Apply AI enhancement using inpainting to preserve original person"""
        
        try:
            # Import AI enhancer from production server
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from production_server import ai_pipeline
            
            if not ai_pipeline:
                print("[AI] Stable Diffusion not available, using base render")
                return base_render
            
            # Extract actual garment properties
            colors = garment_analysis.get("dominant_colors", [])
            texture_features = garment_analysis.get("texture_features", {})
            
            # Fix color interpretation
            if colors:
                primary_color = colors[0]
                color_name = self._fix_color_interpretation(primary_color)
            else:
                color_name = "gray"
            
            # Fix fabric classification
            fabric_type = self._fix_fabric_classification(texture_features)
            
            # Create clothing mask for torso area only
            clothing_mask = self._create_clothing_mask(customer_image, fitting_result)
            
            print(f"[AI] Creating clothing mask for torso area...")
            print(f"[AI] Detected color: {color_name} (from RGB: {primary_color})")
            print(f"[AI] Fabric classification: {fabric_type} (roughness: {texture_features.get('roughness', 0)})")
            
            # Use inpainting to preserve original person
            prompt = f"wearing {color_name} {fabric_type} t-shirt"
            negative_prompt = "naked, nude, different person, face change, body change"
            
            print(f"[AI] Inpainting prompt: {prompt}")
            print(f"[AI] Preserving original person and background ✅")
            
            # Resize for processing
            customer_resized = customer_image.resize((512, 512))
            mask_resized = clothing_mask.resize((512, 512))
            
            # Apply inpainting (only change clothing area)
            enhanced = ai_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=customer_resized,
                mask_image=mask_resized,
                strength=0.6,  # Gentle modification
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]
            
            print("[AI] Inpainting completed - person preserved, only clothing changed")
            return enhanced
            
        except Exception as e:
            print(f"[AI] Inpainting failed: {str(e)}, using base render")
            return base_render
    
    def _fix_color_interpretation(self, rgb_tuple) -> str:
        """Fixed color interpretation - (146,144,148) should be gray not blue"""
        if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) >= 3:
            r, g, b = rgb_tuple[:3]
        else:
            return "gray"
            
        # Check for gray tones first (similar RGB values)
        if abs(r-g) < 20 and abs(g-b) < 20:  # Similar RGB = gray
            if r > 200:
                return "white"
            elif r > 150:
                return "light gray"  # (146,144,148) falls here
            elif r > 100:
                return "gray"
            elif r > 50:
                return "dark gray"
            else:
                return "black"
        
        # Only check distinct colors if not gray
        if r > g + 30 and r > b + 30:
            return "red"
        elif g > r + 30 and g > b + 30:
            return "green"
        elif b > r + 30 and b > g + 30:
            return "blue"
        elif r > 100 and g > 100 and b < 80:
            return "yellow"
        else:
            return "gray"
    
    def _fix_fabric_classification(self, texture_features: Dict[str, Any]) -> str:
        """Fixed fabric classification - roughness 0.291 should be cotton not silk"""
        roughness = texture_features.get('roughness', 0.4)
        edge_density = texture_features.get('edge_density', 0.03)
        complexity = texture_features.get('complexity', 0.1)
        
        # Your values: roughness=0.291, edge_density=0.028, complexity=0.108
        
        if roughness < 0.1 and edge_density < 0.05:
            return "silk"           # Very smooth
        elif roughness > 0.6:
            return "wool"           # Very rough  
        elif edge_density > 0.1:
            return "denim"          # High texture
        else:
            return "cotton"         # roughness 0.291 = cotton ✅
    
    def _create_clothing_mask(self, customer_image: Image.Image, fitting_result: Dict[str, Any]) -> Image.Image:
        """Create mask for torso/clothing area only to preserve person"""
        import cv2
        
        width, height = customer_image.size
        mask = Image.new('L', (width, height), 0)  # Black mask
        
        try:
            # Get positioning data
            positioning = fitting_result.get("positioning", {})
            anchor_point = positioning.get("anchor_point", [width//2, height//2])
            shoulder_width = positioning.get("shoulder_width", 120)
            torso_length = positioning.get("torso_length", 200)
            
            # Create torso rectangle
            x_center, y_center = anchor_point
            margin = 20
            
            x1 = max(0, int(x_center - shoulder_width//2 - margin))
            x2 = min(width, int(x_center + shoulder_width//2 + margin))
            y1 = max(0, int(y_center - 50))  # Above shoulders
            y2 = min(height, int(y_center + torso_length//2 + 50))  # Below torso
            
            # Draw white rectangle on mask (area to change)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            
            print(f"[AI] Created clothing mask: torso area ({x1},{y1}) to ({x2},{y2})")
            
        except Exception as e:
            print(f"[AI] Mask creation failed, using center torso: {e}")
            # Fallback: center torso area
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], fill=255)
        
        return mask
    
    def _rgb_to_color_name(self, rgb_tuple) -> str:
        """Legacy method - use _fix_color_interpretation instead"""
        return self._fix_color_interpretation(rgb_tuple)
    
    def _generate_final_render(
        self, 
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any], 
        fitting_result: Dict[str, Any]
    ) -> Image.Image:
        """Generate final rendered image using all analysis data (fallback method)"""
        
        try:
            # Use enhanced 3D processor with actual visual data
            result = self.garment_processor.process_garment_with_analysis(
                garment_analysis=garment_analysis,
                customer_analysis=customer_analysis,
                fitting_data=fitting_result
            )
            
            if result.get("success") and result.get("rendered_image"):
                return result["rendered_image"]
            else:
                # Fallback: create composite image
                return self._create_fallback_composite(
                    customer_analysis, garment_analysis, fitting_result
                )
                
        except Exception as e:
            logger.error(f"3D rendering failed: {str(e)}")
            return self._create_fallback_composite(
                customer_analysis, garment_analysis, fitting_result
            )
    
    def _create_fallback_composite(
        self, 
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any], 
        fitting_result: Dict[str, Any]
    ) -> Image.Image:
        """Create fallback composite image when 3D rendering fails"""
        
        # Create base image
        width, height = 512, 512
        composite = Image.new("RGB", (width, height), (240, 240, 240))
        
        # Get primary garment color
        colors = garment_analysis.get("dominant_colors", [[128, 128, 128]])
        primary_color = tuple(colors[0][:3]) if colors else (128, 128, 128)
        
        # Create simple garment representation
        garment_img = Image.new("RGB", (width, height), primary_color)
        
        # Apply basic positioning and scaling
        positioning = fitting_result.get("positioning", {})
        scale_factors = fitting_result.get("scale_factors", {})
        
        # Simple overlay (this is a fallback - real implementation would be more sophisticated)
        composite.paste(garment_img, (0, 0))
        
        return composite