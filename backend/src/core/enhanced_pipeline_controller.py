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
            
            # Step 6: Generate final render
            final_image = self._generate_final_render(
                customer_analysis, garment_analysis, fitting_result
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
    
    def _generate_final_render(
        self, 
        customer_analysis: Dict[str, Any], 
        garment_analysis: Dict[str, Any], 
        fitting_result: Dict[str, Any]
    ) -> Image.Image:
        """Generate final rendered image using all analysis data"""
        
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