import os

DISABLE_AI_FOR_DEBUGGING = True
EMERGENCY_BYPASS = True
DISABLE_AI = os.environ.get('DISABLE_AI', 'false').lower() == 'true'
import logging
from typing import Any, Dict
import shutil

import cv2
import numpy as np
from PIL import Image

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
        self._try_load_models()
    
    def _try_load_models(self):
        """Try to load Stable Diffusion models with error handling"""
        try:
            logger.info("🤖 Attempting to load Stable Diffusion models...")
            
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            
            # Check CUDA availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Try to load model
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            
            self.models_loaded = True
            logger.info("✅ Stable Diffusion models loaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Required libraries not available: {e}")
            self.models_loaded = False
        except Exception as e:
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
            logger.info("⚠️ Stable Diffusion not available, returning original image")
            return rendered_image

        try:
            photo_style = self._analyze_photo_style(original_photo)
            prompt = self._create_enhancement_prompt(photo_style)
            negative_prompt = self._create_negative_prompt()

            enhanced = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=rendered_image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50,
            ).images[0]

            logger.info("✅ AI enhancement completed")
            return enhanced

        except Exception as e:
            logger.error(f"⚠️ AI enhancement failed: {e}")
            return rendered_image

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

    def _create_negative_prompt(self) -> str:
        """Create negative prompt for better quality"""
        return (
            "cartoon, illustration, painting, low quality, blurry, distorted, "
            "deformed, "
            "artificial, plastic, fake, unrealistic, bad anatomy, bad proportions, "
            "extra limbs, missing limbs, floating limbs, disconnected limbs"
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
