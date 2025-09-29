import os
import logging
from typing import Dict, Optional
import numpy as np
from .vertex_ai_tryon import VertexAITryOn

logger = logging.getLogger(__name__)

class FAHNTryOn:
    """FAHN API Virtual Try-On integration (fallback service)"""
    
    def __init__(self):
        self.api_key = os.getenv("FAHN_API_KEY")
        self.base_url = os.getenv("FAHN_BASE_URL", "https://api.fahn.ai/v1")
        
        if not self.api_key:
            logger.warning("FAHN_API_KEY not configured")
    
    async def virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray, 
                           customer_analysis: Dict, garment_analysis: Dict,
                           product_info: Dict) -> Dict:
        """
        Process virtual try-on using FAHN API
        """
        try:
            if not self.api_key:
                raise Exception("FAHN API key not configured")
            
            logger.info("[FAHN] Starting virtual try-on processing")
            
            # Convert images to base64
            customer_b64 = self._image_to_base64(customer_image)
            garment_b64 = self._image_to_base64(garment_image)
            
            # Prepare FAHN API request
            import aiohttp
            
            payload = {
                "model": "virtual-tryon-v1",
                "customer_image": customer_b64,
                "garment_image": garment_b64,
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
        import cv2
        import base64
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Convert to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def is_available(self) -> bool:
        """Check if FAHN service is available"""
        return bool(self.api_key)


class VirtualTryOnFallbackService:
    """
    Fallback service that tries: FAHN → Vertex AI → Error Message
    FAHN is primary, Vertex AI is fallback
    """
    
    def __init__(self):
        self.vertex_ai = None
        self.fahn = FAHNTryOn()
        
        # Initialize Vertex AI if configured
        try:
            self.vertex_ai = VertexAITryOn()
        except Exception as e:
            logger.warning(f"Vertex AI not available: {e}")
    
    async def process_virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray,
                                  customer_analysis: Dict, garment_analysis: Dict,
                                  product_info: Dict) -> Dict:
        """
        Process virtual try-on with fallback chain:
        1. FAHN (primary)
        2. Vertex AI (fallback - when available)
        3. Error message (final fallback)
        """
        
        # Try FAHN first (primary)
        if self.fahn.is_available():
            logger.info("[FALLBACK] Trying FAHN (primary)")
            result = await self.fahn.virtual_tryon(
                customer_image, garment_image, customer_analysis,
                garment_analysis, product_info
            )
            
            if result.get("success"):
                logger.info("[FALLBACK] ✅ FAHN succeeded")
                return result
            else:
                logger.warning(f"[FALLBACK] FAHN failed: {result.get('error')}")
        
        # Try Vertex AI as fallback (when available)
        if self.vertex_ai and self.vertex_ai.is_available():
            logger.info("[FALLBACK] Trying Vertex AI (fallback)")
            result = await self.vertex_ai.virtual_tryon(
                customer_image, garment_image, customer_analysis, 
                garment_analysis, product_info
            )
            
            if result.get("success"):
                logger.info("[FALLBACK] ✅ Vertex AI succeeded")
                return result
            else:
                logger.warning(f"[FALLBACK] Vertex AI failed: {result.get('error')}")
        
        # Final fallback - return original image with error message
        logger.error("[FALLBACK] All services failed - returning error")
        return {
            "success": False,
            "result_image_base64": self._image_to_base64(customer_image),  # Return original
            "service_used": "none",
            "error_message": "Virtual try-on is not available at the moment. Please contact support.",
            "user_message": "Virtual try-on service is temporarily unavailable. Please try again later or contact our support team for assistance."
        }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        import cv2
        import base64
        
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')