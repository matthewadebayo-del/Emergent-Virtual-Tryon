import os
import base64
import logging
import aiohttp
import asyncio
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class FASHNTryOn:
    """FASHN Virtual Try-On API integration (fallback service)"""
    
    def __init__(self):
        self.api_key = os.getenv("FASHN_API_KEY")
        self.base_url = os.getenv("FASHN_BASE_URL", "https://api.fashn.ai/v1")
        
        if not self.api_key:
            logger.warning("FASHN API key not configured - service unavailable")
    
    async def virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray,
                           customer_analysis: Dict, garment_analysis: Dict,
                           product_info: Dict) -> Dict:
        """
        Process virtual try-on using FASHN API
        """
        try:
            logger.info("[FASHN] Starting virtual try-on processing")
            
            if not self.api_key:
                raise Exception("FASHN API key not configured")
            
            # Convert images to base64
            customer_b64 = self._image_to_base64(customer_image)
            garment_b64 = self._image_to_base64(garment_image)
            
            # Prepare request payload using correct FASHN API format
            payload = {
                "model_name": "tryon-v1.5",
                "inputs": {
                    "model_image": customer_b64,
                    "garment_image": garment_b64,
                    "category": self._map_category(product_info.get("category", "tops")),
                    "moderation_level": "strict"
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make async request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/run",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Handle new response format
                        output_image = None
                        if "outputs" in result and result["outputs"]:
                            output_image = result["outputs"][0].get("image")
                        elif "output" in result:
                            output_image = result["output"].get("image")
                        
                        return {
                            "success": True,
                            "result_image_base64": output_image,
                            "service_used": "fashn",
                            "processing_time": result.get("processing_time", 0)
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"FASHN API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"[FASHN] Processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"FASHN processing failed: {str(e)}",
                "service_used": "fashn"
            }
    
    def _map_category(self, category: str) -> str:
        """Map product category to FASHN API category"""
        category_mapping = {
            "shirts": "tops",
            "mens_shirts": "tops",
            "womens_shirts": "tops",
            "pants": "bottoms",
            "jeans": "bottoms",
            "dresses": "dresses",
            "jackets": "outerwear",
            "blazers": "outerwear"
        }
        return category_mapping.get(category.lower(), "tops")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        import cv2
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Convert to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def is_available(self) -> bool:
        """Check if FASHN service is available"""
        return bool(self.api_key)