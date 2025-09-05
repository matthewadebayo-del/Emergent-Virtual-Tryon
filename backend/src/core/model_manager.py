import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager with lazy loading to prevent startup timeouts"""

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @lru_cache(maxsize=None)
    def get_stable_diffusion_model(self):
        """Lazy load SD model - only when first needed"""
        if "sd_img2img" not in self._models:
            logger.info("Loading Stable Diffusion model on-demand...")
            start_time = time.time()

            try:
                import torch
                from diffusers import StableDiffusionImg2ImgPipeline

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32

                self._models["sd_img2img"] = (
                    StableDiffusionImg2ImgPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-1",
                        torch_dtype=dtype,
                        cache_dir=os.getenv("MODEL_CACHE_DIR", "/root/.cache"),
                        safety_checker=None,
                        requires_safety_checker=False,
                    ).to(device)
                )

                load_time = time.time() - start_time
                logger.info(
                    f"✅ SD model loaded successfully in {load_time:.2f}s on {device}"
                )

            except Exception as e:
                logger.error(f"❌ Failed to load SD model: {e}")
                self._models["sd_img2img"] = None

        return self._models["sd_img2img"]

    @lru_cache(maxsize=None)
    def get_body_reconstructor(self):
        """Lazy load body reconstruction model"""
        if "body_reconstructor" not in self._models:
            logger.info("Loading body reconstruction model on-demand...")
            start_time = time.time()

            try:
                from src.core.body_reconstruction import BodyReconstructor

                self._models["body_reconstructor"] = BodyReconstructor()

                load_time = time.time() - start_time
                logger.info(f"✅ Body reconstructor loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load body reconstructor: {e}")
                self._models["body_reconstructor"] = None

        return self._models["body_reconstructor"]

    @lru_cache(maxsize=None)
    def get_garment_fitter(self):
        """Lazy load garment fitting model"""
        if "garment_fitter" not in self._models:
            logger.info("Loading garment fitting model on-demand...")
            start_time = time.time()

            try:
                from src.core.garment_fitting import GarmentFitter

                self._models["garment_fitter"] = GarmentFitter()

                load_time = time.time() - start_time
                logger.info(f"✅ Garment fitter loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load garment fitter: {e}")
                self._models["garment_fitter"] = None

        return self._models["garment_fitter"]

    @lru_cache(maxsize=None)
    def get_renderer(self):
        """Lazy load 3D renderer"""
        if "renderer" not in self._models:
            logger.info("Loading 3D renderer on-demand...")
            start_time = time.time()

            try:
                from src.core.rendering import PhotorealisticRenderer

                self._models["renderer"] = PhotorealisticRenderer()

                load_time = time.time() - start_time
                logger.info(f"✅ 3D renderer loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load 3D renderer: {e}")
                self._models["renderer"] = None

        return self._models["renderer"]

    @lru_cache(maxsize=None)
    def get_ai_enhancer(self):
        """Lazy load AI enhancement model"""
        if "ai_enhancer" not in self._models:
            logger.info("Loading AI enhancement model on-demand...")
            start_time = time.time()

            try:
                from src.core.ai_enhancement import AIEnhancer

                self._models["ai_enhancer"] = AIEnhancer()

                load_time = time.time() - start_time
                logger.info(f"✅ AI enhancer loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load AI enhancer: {e}")
                self._models["ai_enhancer"] = None

        return self._models["ai_enhancer"]

    def warmup_essential_models(self):
        """Pre-load only the most critical models during startup"""
        logger.info("🔥 Warming up essential models for fast startup...")
        start_time = time.time()

        try:
            self.get_body_reconstructor()

            warmup_time = time.time() - start_time
            logger.info(f"✅ Essential models warmed up in {warmup_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models for health checks"""
        return {
            "models_loaded": len(self._models),
            "available_models": list(self._models.keys()),
            "sd_model_loaded": "sd_img2img" in self._models
            and self._models["sd_img2img"] is not None,
            "body_reconstructor_loaded": "body_reconstructor" in self._models
            and self._models["body_reconstructor"] is not None,
            "garment_fitter_loaded": "garment_fitter" in self._models
            and self._models["garment_fitter"] is not None,
            "renderer_loaded": "renderer" in self._models
            and self._models["renderer"] is not None,
            "ai_enhancer_loaded": "ai_enhancer" in self._models
            and self._models["ai_enhancer"] is not None,
        }


model_manager = ModelManager()
