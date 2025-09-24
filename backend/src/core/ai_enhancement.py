import os

DISABLE_AI_FOR_DEBUGGING = False
EMERGENCY_BYPASS = False
DISABLE_AI = os.environ.get('DISABLE_AI', 'false').lower() == 'true'
import logging
from typing import Any, Dict
import shutil

import cv2
import numpy as np
from PIL import Image, ImageFilter

if DISABLE_AI or DISABLE_AI_FOR_DEBUGGING:
    AI_ENHANCEMENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("🔧 Step 6: AI enhancement disabled - using fallback mode")
else:
    try:
        import torch
        print(f"✅ torch imported successfully: {torch.__version__}")
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
        print(f"✅ diffusers imported successfully")
        AI_ENHANCEMENT_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ AI enhancement dependencies not available: {e}")
        import traceback
        traceback.print_exc()
        AI_ENHANCEMENT_AVAILABLE = False

    logger = logging.getLogger(__name__)


class FixedAIEnhancer:
    """Fixed AI enhancer with proper error handling"""
    
    def __init__(self):
        if DISABLE_AI:
            self.models_loaded = False
            print("🔧 Step 6: AI disabled via runtime environment variable")
            return
        if EMERGENCY_BYPASS:
            self.models_loaded = False
            print("AI BYPASSED - focusing on 3D rendering")
            return
        if DISABLE_AI_FOR_DEBUGGING:
            self.models_loaded = False
            logger.info("🔧 AI temporarily disabled for 3D debugging")
            return
            
        self.models_loaded = False
        self.pipe = None
        # Don't load models immediately - load on first use to save startup time
        print("[AI] AI Enhancer initialized - models will load on first use")
    
    def _try_load_models(self):
        """Try to load Stable Diffusion models with error handling"""
        try:
            print("[AI] Attempting to load Stable Diffusion models...")
            logger.info("🤖 Attempting to load Stable Diffusion models...")
            
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            
            # Check CUDA availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[AI] Using device: {device}")
            logger.info(f"Using device: {device}")
            
            # Try to load model with reduced memory usage
            print("[AI] Loading Stable Diffusion pipeline...")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",  # Use smaller, more reliable model
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            ).to(device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
            
            self.models_loaded = True
            print("[AI] ✅ Stable Diffusion models loaded successfully")
            logger.info("✅ Stable Diffusion models loaded successfully")
            
        except ImportError as e:
            print(f"[AI] ⚠️ Required libraries not available: {e}")
            logger.warning(f"⚠️ Required libraries not available: {e}")
            self.models_loaded = False
        except Exception as e:
            print(f"[AI] ⚠️ Could not load Stable Diffusion models: {e}")
            logger.warning(f"⚠️ Could not load Stable Diffusion models: {e}")
            self.models_loaded = False
    
    def enhance_image(self, image_path: str, output_path: str) -> bool:
        """Enhance image with AI or return original"""
        if EMERGENCY_BYPASS:
            import shutil
            shutil.copy2(image_path, output_path)
            return True
        if DISABLE_AI_FOR_DEBUGGING:
            shutil.copy2(image_path, output_path)
            logger.info(f"🔧 AI bypass: copied {image_path} to {output_path}")
            return True
            
        try:
            if not self.models_loaded:
                logger.info("📋 Stable Diffusion not available, copying original image")
                # Just copy the original file
                shutil.copy2(image_path, output_path)
                return True
            
            # Load image
            image = Image.open(image_path)
            
            # Enhance with Stable Diffusion
            prompt = "photorealistic, high quality, detailed clothing, natural lighting"
            enhanced = self.pipe(
                prompt=prompt,
                image=image,
                strength=0.2,  # Light enhancement
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]
            
            enhanced.save(output_path)
            logger.info(f"✅ AI enhancement complete: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ AI enhancement failed: {e}")
            # Fallback to copying original
            try:
                shutil.copy2(image_path, output_path)
                logger.info("📋 Copied original image as fallback")
                return True
            except:
                return False

    def enhance_realism(
        self,
        rendered_image: Image.Image,
        original_photo: Image.Image,
        strength: float = 0.3,
    ) -> Image.Image:
        """Enhance rendered image to match original photo style - compatibility method"""
        if not self.models_loaded:
            logger.info("⚠️ Stable Diffusion not available, creating basic virtual try-on")
            return self._create_basic_virtual_tryon(original_photo, rendered_image)

        try:
            # Use the original photo as base and apply garment from rendered image
            photo_style = self._analyze_photo_style(original_photo)
            
            # Create a prompt that specifically mentions adding clothing to the person
            garment_prompt = self._create_garment_application_prompt(photo_style)
            negative_prompt = self._create_negative_prompt()

            # Use original photo as base with garment application prompt
            enhanced = self.pipe(
                prompt=garment_prompt,
                negative_prompt=negative_prompt,
                image=original_photo,  # Use original photo as base
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=20,  # Reduced for faster processing
            ).images[0]

            logger.info("✅ AI garment application completed")
            return enhanced

        except Exception as e:
            logger.error(f"⚠️ AI enhancement failed: {e}")
            # Fallback to basic virtual try-on
            return self._create_basic_virtual_tryon(original_photo, rendered_image)

    def _analyze_photo_style(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze original photo for style characteristics"""
        img_array = np.array(image)

        brightness = np.mean(img_array)
        contrast = np.std(img_array)

        r_mean = np.mean(img_array[:, :, 0])
        b_mean = np.mean(img_array[:, :, 2])
        color_temp = "warm" if r_mean > b_mean else "cool"

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        background_complexity = np.sum(edges) / (
            img_array.shape[0] * img_array.shape[1]
        )

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])

        return {
            "brightness": brightness,
            "contrast": contrast,
            "color_temperature": color_temp,
            "background_complexity": background_complexity,
            "saturation": saturation,
        }

    def _create_enhancement_prompt(self, style_info: Dict[str, Any]) -> str:
        """Create enhancement prompt based on style analysis"""
        base_prompt = (
            "photorealistic, high quality, detailed, natural lighting, "
            "realistic skin texture, professional photography"
        )

        if style_info["brightness"] > 150:
            base_prompt += ", bright lighting, well-lit"
        elif style_info["brightness"] < 100:
            base_prompt += ", moody lighting, dramatic shadows"

        if style_info["color_temperature"] == "warm":
            base_prompt += ", warm tones, golden hour lighting"
        else:
            base_prompt += ", cool tones, natural daylight"

        if style_info["background_complexity"] < 0.1:
            base_prompt += ", simple background, clean backdrop"
        else:
            base_prompt += ", detailed background"

        if style_info["saturation"] > 100:
            base_prompt += ", vibrant colors"
        else:
            base_prompt += ", natural colors"

        return base_prompt
    
    def _create_garment_application_prompt(self, style_info: Dict[str, Any]) -> str:
        """Create prompt specifically for applying garment to person"""
        base_prompt = (
            "person wearing a classic white t-shirt, photorealistic, high quality, "
            "detailed clothing texture, natural fit, realistic fabric, "
            "professional photography, natural lighting"
        )

        if style_info["brightness"] > 150:
            base_prompt += ", bright lighting, well-lit"
        elif style_info["brightness"] < 100:
            base_prompt += ", moody lighting, dramatic shadows"

        if style_info["color_temperature"] == "warm":
            base_prompt += ", warm tones, golden hour lighting"
        else:
            base_prompt += ", cool tones, natural daylight"

        return base_prompt
    
    def enhance_realism_with_garment(
        self,
        rendered_image: Image.Image,
        original_photo: Image.Image,
        clothing_description: str,
        strength: float = 0.3,
    ) -> Image.Image:
        """Enhanced method using 3D rendered image as reference guide"""
        # Try to load models if not already loaded
        if not self.models_loaded and self.pipe is None:
            print("[AI] Loading models on first use...")
            self._try_load_models()
        
        if not self.models_loaded:
            print("[AI] Stable Diffusion not available, using 3D-guided fallback")
            logger.info("⚠️ Stable Diffusion not available, creating basic virtual try-on")
            return self._blend_3d_with_original(rendered_image, original_photo)

        try:
            # Use 3D rendered image as reference to guide garment application
            photo_style = self._analyze_photo_style(original_photo)
            garment_prompt = self._create_3d_guided_prompt(photo_style, clothing_description)
            negative_prompt = self._create_negative_prompt()

            print(f"[AI] Enhancing image realism with {clothing_description}")
            print(f"[AI] Using 3D-guided enhancement with strength: {strength}")
            logger.info(f"[AI] Enhancing image realism with {clothing_description}")
            logger.info(f"[AI] Using strength: {strength}, guidance: 7.5, steps: 20 (pose-preserving)")

            # First pass: Apply 3D garment positioning to original photo
            enhanced = self.pipe(
                prompt=garment_prompt,
                negative_prompt=negative_prompt,
                image=original_photo,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=20,
            ).images[0]

            # Second pass: Refine using 3D rendered image as reference
            refined = self.pipe(
                prompt=f"photorealistic person wearing {clothing_description}, natural lighting, high quality",
                negative_prompt=negative_prompt,
                image=enhanced,
                strength=0.15,  # Light refinement
                guidance_scale=7.5,
                num_inference_steps=10,
            ).images[0]

            # Blend with 3D positioning information
            final_result = self._blend_with_3d_reference(refined, rendered_image, original_photo)

            print("[AI] 3D-guided enhancement completed")
            logger.info("✅ AI garment application with 3D guidance completed")
            return final_result

        except Exception as e:
            logger.error(f"⚠️ AI enhancement failed: {e}")
            return self._blend_3d_with_original(rendered_image, original_photo)
    
    def _create_3d_guided_prompt(self, style_info: Dict[str, Any], clothing_description: str) -> str:
        """Create prompt that leverages 3D positioning information"""
        garment_type = "t-shirt"
        if "polo" in clothing_description.lower():
            garment_type = "polo shirt"
        elif "dress shirt" in clothing_description.lower():
            garment_type = "dress shirt"
        elif "jean" in clothing_description.lower():
            garment_type = "jeans"
        elif "blazer" in clothing_description.lower():
            garment_type = "blazer"
        
        color = "white"
        if "navy" in clothing_description.lower():
            color = "navy blue"
        elif "black" in clothing_description.lower():
            color = "black"
        elif "blue" in clothing_description.lower():
            color = "blue"
        
        return (
            f"person wearing a perfectly fitted {color} {garment_type}, "
            f"realistic fabric draping, natural shadows, photorealistic, "
            f"high quality, detailed clothing texture, professional photography"
        )
    
    def _blend_with_3d_reference(self, ai_result: Image.Image, rendered_3d: Image.Image, original_photo: Image.Image) -> Image.Image:
        """Blend AI result with 3D positioning reference"""
        try:
            # Extract garment mask from 3D rendered image
            garment_mask = self._extract_garment_mask(rendered_3d)
            
            # Blend AI result with original photo using 3D mask guidance
            if garment_mask is not None:
                # Use 3D mask to guide blending
                result = Image.composite(ai_result, original_photo, garment_mask)
                return result
            else:
                return ai_result
                
        except Exception as e:
            logger.warning(f"3D reference blending failed: {e}")
            return ai_result
    
    def _extract_garment_mask(self, rendered_image: Image.Image) -> Image.Image:
        """Extract garment area from 3D rendered image"""
        try:
            import numpy as np
            
            # Convert to numpy array
            img_array = np.array(rendered_image)
            
            # Create mask based on non-background areas
            # Assume background is light colored (> 200 in all channels)
            background_mask = np.all(img_array > 200, axis=2)
            garment_mask = ~background_mask
            
            # Convert back to PIL image
            mask_array = (garment_mask * 255).astype(np.uint8)
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            logger.warning(f"Garment mask extraction failed: {e}")
            return None
    
    def _blend_3d_with_original(self, rendered_image: Image.Image, original_photo: Image.Image) -> Image.Image:
        """Fallback: Blend 3D rendered image with original photo"""
        try:
            print("[AI] Creating 3D-guided fallback blend")
            
            # Resize rendered image to match original if needed
            if rendered_image.size != original_photo.size:
                rendered_image = rendered_image.resize(original_photo.size, Image.Resampling.LANCZOS)
            
            # Convert images to RGBA for blending
            original_rgba = original_photo.convert('RGBA')
            rendered_rgba = rendered_image.convert('RGBA')
            
            # Create a simple blend - use rendered image for torso area
            width, height = original_photo.size
            
            # Create mask for torso area where garment should be
            from PIL import ImageDraw
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            # Define torso area
            torso_top = int(height * 0.2)
            torso_bottom = int(height * 0.7)
            torso_left = int(width * 0.25)
            torso_right = int(width * 0.75)
            
            # Draw elliptical mask for torso
            draw.ellipse([torso_left, torso_top, torso_right, torso_bottom], fill=255)
            
            # Apply gaussian blur to soften edges
            mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
            
            # Blend images using the mask
            result = Image.composite(rendered_rgba, original_rgba, mask)
            
            print("[AI] 3D-guided fallback blend completed")
            logger.info("✅ 3D-guided fallback blending completed")
            return result.convert('RGB')
                
        except Exception as e:
            print(f"[AI] 3D blending failed: {e}")
            logger.error(f"3D blending failed: {e}")
            # Final fallback - create visible garment overlay
            return self._create_basic_virtual_tryon_with_description(original_photo, "white t-shirt")
    
    def _create_specific_garment_prompt(self, style_info: Dict[str, Any], clothing_description: str) -> str:
        """Create prompt specifically for the described garment"""
        # Extract garment type from description
        garment_type = "t-shirt"  # default
        if "polo" in clothing_description.lower():
            garment_type = "polo shirt"
        elif "dress shirt" in clothing_description.lower():
            garment_type = "dress shirt"
        elif "jean" in clothing_description.lower():
            garment_type = "jeans"
        elif "chino" in clothing_description.lower():
            garment_type = "chino pants"
        elif "blazer" in clothing_description.lower():
            garment_type = "blazer"
        elif "dress" in clothing_description.lower():
            garment_type = "dress"
        
        # Extract color from description
        color = "white"  # default
        if "navy" in clothing_description.lower():
            color = "navy blue"
        elif "black" in clothing_description.lower():
            color = "black"
        elif "blue" in clothing_description.lower():
            color = "blue"
        elif "khaki" in clothing_description.lower():
            color = "khaki"
        
        base_prompt = (
            f"person wearing a {color} {garment_type}, photorealistic, high quality, "
            f"detailed clothing texture, natural fit, realistic fabric, "
            f"professional photography, natural lighting, well-fitted clothing"
        )

        if style_info["brightness"] > 150:
            base_prompt += ", bright lighting, well-lit"
        elif style_info["brightness"] < 100:
            base_prompt += ", moody lighting, dramatic shadows"

        if style_info["color_temperature"] == "warm":
            base_prompt += ", warm tones, golden hour lighting"
        else:
            base_prompt += ", cool tones, natural daylight"

        return base_prompt
    
    def _create_basic_virtual_tryon_with_description(self, original_photo: Image.Image, clothing_description: str) -> Image.Image:
        """Create basic virtual try-on with specific clothing description"""
        try:
            from PIL import ImageDraw, ImageFilter, ImageEnhance
            import numpy as np
            
            # Convert to RGB if needed
            if original_photo.mode != 'RGB':
                original_photo = original_photo.convert('RGB')
            
            # Create a copy to work with
            result_image = original_photo.copy()
            width, height = result_image.size
            
            # Create clothing overlay
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Estimate torso area (simplified)
            torso_top = int(height * 0.25)  # Approximate shoulder area
            torso_bottom = int(height * 0.65)  # Approximate waist area
            torso_left = int(width * 0.3)
            torso_right = int(width * 0.7)
            
            # Determine color from description
            if "navy" in clothing_description.lower():
                color_rgb = (25, 25, 112)  # Navy blue
            elif "black" in clothing_description.lower():
                color_rgb = (40, 40, 40)  # Black
            elif "blue" in clothing_description.lower():
                color_rgb = (70, 130, 180)  # Steel blue
            elif "khaki" in clothing_description.lower():
                color_rgb = (195, 176, 145)  # Khaki
            else:
                color_rgb = (255, 255, 255)  # White (default)
            
            # Draw clothing based on type
            if "shirt" in clothing_description.lower() or "polo" in clothing_description.lower():
                # Draw shirt shape
                draw.rectangle(
                    [torso_left, torso_top, torso_right, torso_bottom],
                    fill=color_rgb + (180,)  # Semi-transparent
                )
                
                # Add sleeves
                sleeve_width = int(width * 0.12)
                sleeve_height = int((torso_bottom - torso_top) * 0.6)
                draw.rectangle(
                    [torso_left - sleeve_width, torso_top, torso_left, torso_top + sleeve_height],
                    fill=color_rgb + (160,)
                )
                draw.rectangle(
                    [torso_right, torso_top, torso_right + sleeve_width, torso_top + sleeve_height],
                    fill=color_rgb + (160,)
                )
            
            # Apply overlay with blending
            result_image = Image.alpha_composite(
                result_image.convert('RGBA'), overlay
            ).convert('RGB')
            
            # Enhance the result
            enhancer = ImageEnhance.Contrast(result_image)
            result_image = enhancer.enhance(1.05)
            
            logger.info(f"✅ Basic virtual try-on created for {clothing_description}")
            return result_image
            
        except Exception as e:
            logger.error(f"Basic virtual try-on failed: {e}")
            return original_photo
    
    def _create_basic_virtual_tryon(self, original_photo: Image.Image, rendered_image: Image.Image) -> Image.Image:
        """Create basic virtual try-on when AI is not available"""
        try:
            from PIL import ImageDraw, ImageFilter, ImageEnhance
            import numpy as np
            
            # Convert to RGB if needed
            if original_photo.mode != 'RGB':
                original_photo = original_photo.convert('RGB')
            
            # Create a copy to work with
            result_image = original_photo.copy()
            width, height = result_image.size
            
            # Create clothing overlay
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Estimate torso area (simplified)
            torso_top = int(height * 0.25)  # Approximate shoulder area
            torso_bottom = int(height * 0.65)  # Approximate waist area
            torso_left = int(width * 0.3)
            torso_right = int(width * 0.7)
            
            # Draw white t-shirt
            color_rgb = (255, 255, 255)  # White t-shirt
            
            # Draw shirt shape
            draw.rectangle(
                [torso_left, torso_top, torso_right, torso_bottom],
                fill=color_rgb + (180,)  # Semi-transparent white
            )
            
            # Add sleeves
            sleeve_width = int(width * 0.12)
            sleeve_height = int((torso_bottom - torso_top) * 0.6)
            draw.rectangle(
                [torso_left - sleeve_width, torso_top, torso_left, torso_top + sleeve_height],
                fill=color_rgb + (160,)
            )
            draw.rectangle(
                [torso_right, torso_top, torso_right + sleeve_width, torso_top + sleeve_height],
                fill=color_rgb + (160,)
            )
            
            # Apply overlay with blending
            result_image = Image.alpha_composite(
                result_image.convert('RGBA'), overlay
            ).convert('RGB')
            
            # Enhance the result
            enhancer = ImageEnhance.Contrast(result_image)
            result_image = enhancer.enhance(1.05)
            
            logger.info("✅ Basic virtual try-on created")
            return result_image
            
        except Exception as e:
            logger.error(f"Basic virtual try-on failed: {e}")
            return original_photo

    def _create_negative_prompt(self) -> str:
        """Create negative prompt for better quality"""
        return (
            "cartoon, illustration, painting, low quality, blurry, distorted, "
            "deformed, artificial, plastic, fake, unrealistic, bad anatomy, bad proportions, "
            "extra limbs, missing limbs, floating limbs, disconnected limbs, "
            "naked, nude, shirtless, bare chest, no clothing"
        )

    def seamless_blend(
        self,
        enhanced_image: Image.Image,
        original_photo: Image.Image,
        garment_mask: Image.Image,
    ) -> Image.Image:
        """Seamlessly blend enhanced garment with original photo"""
        logger.info("⚠️ Seamless blending not implemented in fixed version, returning enhanced image")
        return enhanced_image


AIEnhancer = FixedAIEnhancer
