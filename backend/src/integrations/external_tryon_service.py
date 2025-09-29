import os
import logging
from typing import Dict, Optional
import numpy as np
import base64
import cv2
import aiohttp

logger = logging.getLogger(__name__)

class FAHNTryOnService:
    """FAHN API Virtual Try-On Service (Primary)"""
    
    def __init__(self):
        self.api_key = os.getenv("FAHN_API_KEY")
        self.base_url = os.getenv("FAHN_BASE_URL", "https://api.fahn.ai/v1")
        
        if not self.api_key:
            logger.warning("FAHN_API_KEY not configured")
    
    async def virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray, 
                           product_info: Dict) -> Dict:
        """Process virtual try-on using FAHN API"""
        try:
            if not self.api_key:
                raise Exception("FAHN API key not configured")
            
            logger.info("[FAHN] Starting virtual try-on processing")
            
            # Convert images to base64
            customer_b64 = self._image_to_base64(customer_image)
            garment_b64 = self._image_to_base64(garment_image)
            
            payload = {
                "model": "virtual-tryon-v1",
                "customer_image": customer_b64,
                "garment_image": garment_b64,
                "product_info": product_info,
                "options": {
                    "preserve_identity": True,
                    "quality": "high"
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/virtual-tryon",
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "success": True,
                            "result_image_base64": result.get("result_image"),
                            "service_used": "fahn",
                            "processing_time": result.get("processing_time", 0)
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"FAHN API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"[FAHN] Processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"FAHN processing failed: {str(e)}",
                "service_used": "fahn"
            }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def is_available(self) -> bool:
        """Check if FAHN service is available"""
        return bool(self.api_key)


class ExternalTryOnService:
    """
    Clean external API service: FAHN → Vertex AI → Error Message
    No built-in computer vision system
    """
    
    def __init__(self):
        self.fahn = FAHNTryOnService()
        # Vertex AI will be added when available
    
    async def process_virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray,
                                  product_info: Dict) -> Dict:
        """
        Process virtual try-on with clean external API chain:
        1. FAHN (primary)
        2. Vertex AI (when available)
        3. Error message (final fallback)
        """
        
        # Try FAHN first (primary)
        if self.fahn.is_available():
            logger.info("[EXTERNAL] Trying FAHN (primary)")
            result = await self.fahn.virtual_tryon(customer_image, garment_image, product_info)
            
            if result.get("success"):
                logger.info("[EXTERNAL] ✅ FAHN succeeded")
                return result
            else:
                logger.warning(f"[EXTERNAL] FAHN failed: {result.get('error')}")
        
        # TODO: Try Vertex AI when available
        # if self.vertex_ai and self.vertex_ai.is_available():
        #     logger.info("[EXTERNAL] Trying Vertex AI (fallback)")
        #     result = await self.vertex_ai.virtual_tryon(...)
        #     if result.get("success"):
        #         return result
        
        # Final fallback - return original image with error message
        logger.error("[EXTERNAL] All services failed - returning error")
        return {
            "success": False,
            "result_image_base64": self._image_to_base64(customer_image),  # Return original
            "service_used": "none",
            "error_message": "Virtual try-on is not available at the moment. Please contact support.",
            "user_message": "Virtual try-on service is temporarily unavailable. Please try again later or contact our support team for assistance."
        }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')