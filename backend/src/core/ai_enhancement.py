from typing import Any, Dict

import cv2
import numpy as np
import torch
from diffusers import (StableDiffusionImg2ImgPipeline,
                       StableDiffusionInpaintPipeline)
from PIL import Image


class AIEnhancer:
    """AI enhancement using Stable Diffusion with advanced style matching"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._load_models()

    def _load_models(self):
        """Load Stable Diffusion models"""
        try:
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)

            try:
                self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                self.inpaint_pipe = self.inpaint_pipe.to(self.device)
                print(f"✅ Stable Diffusion pipelines loaded on {self.device}")
            except Exception as e:
                print(f"⚠️ Inpainting pipeline failed to load: {e}")
                self.inpaint_pipe = None

        except Exception as e:
            print(f"⚠️ Failed to load Stable Diffusion: {e}")
            self.img2img_pipe = None
            self.inpaint_pipe = None

    def enhance_realism(
        self,
        rendered_image: Image.Image,
        original_photo: Image.Image,
        strength: float = 0.3,
    ) -> Image.Image:
        """Enhance rendered image to match original photo style"""
        if self.img2img_pipe is None:
            print(
                "⚠️ Stable Diffusion not available, returning original rendered image"
            )
            return rendered_image

        try:
            photo_style = self._analyze_photo_style(original_photo)
            prompt = self._create_enhancement_prompt(photo_style)
            negative_prompt = self._create_negative_prompt()

            enhanced = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=rendered_image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50,
            ).images[0]

            print("✅ AI enhancement completed")
            return enhanced

        except Exception as e:
            print(f"⚠️ AI enhancement failed: {e}")
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
        if self.inpaint_pipe is None:
            print("⚠️ Inpainting not available, returning enhanced image")
            return enhanced_image

        try:
            blended = self.inpaint_pipe(
                prompt="seamless integration, natural lighting, photorealistic",
                image=original_photo,
                mask_image=garment_mask,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images[0]

            return blended

        except Exception as e:
            print(f"⚠️ Seamless blending failed: {e}")
            return enhanced_image
