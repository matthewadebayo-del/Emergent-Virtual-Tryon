import os
import logging
from typing import Dict, Optional
import numpy as np
import base64
import cv2
import aiohttp

logger = logging.getLogger(__name__)

class FASHNTryOnService:
    """FASHN API Virtual Try-On Service (Primary)"""
    
    def __init__(self):
        self.api_key = os.getenv("FASHN_API_KEY", os.getenv("FAHN_API_KEY"))  # Support both names
        self.base_url = "https://api.fashn.ai/v1"
        
        if not self.api_key:
            logger.warning("FASHN_API_KEY not configured")
    
    async def virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray, 
                           product_info: Dict) -> Dict:
        """Process virtual try-on using FASHN API"""
        try:
            if not self.api_key:
                raise Exception("FASHN API key not configured")
            
            logger.info("[FASHN] Starting virtual try-on processing")
            
            # Convert images to base64
            customer_b64 = self._image_to_base64(customer_image)
            garment_b64 = self._image_to_base64(garment_image)
            
            # FASHN API payload format
            payload = {
                "model_name": "product-to-model",
                "inputs": {
                    "product_image": f"data:image/jpeg;base64,{garment_b64}",
                    "model_image": f"data:image/jpeg;base64,{customer_b64}",
                    "output_format": "png",
                    "return_base64": True
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Submit job
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/run",
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        job_result = await response.json()
                        job_id = job_result.get("id")
                        
                        if not job_id:
                            raise Exception("No job ID returned")
                        
                        # Poll for completion
                        result = await self._poll_job_status(session, job_id, headers)
                        return result
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
    
    async def _poll_job_status(self, session, job_id: str, headers: Dict, max_attempts: int = 30) -> Dict:
        """Poll FASHN job status until completion"""
        import asyncio
        
        for attempt in range(max_attempts):
            try:
                async with session.get(
                    f"{self.base_url}/status/{job_id}",
                    headers=headers,
                    timeout=10
                ) as response:
                    
                    if response.status == 200:
                        status_result = await response.json()
                        status = status_result.get("status")
                        
                        if status == "completed":
                            output = status_result.get("output", [])
                            if output and len(output) > 0:
                                # Extract base64 from data URL or use CDN URL
                                result_image = output[0]
                                if result_image.startswith("data:image"):
                                    result_b64 = result_image.split(",")[1]
                                else:
                                    # Download from CDN URL and convert to base64
                                    async with session.get(result_image) as img_response:
                                        img_bytes = await img_response.read()
                                        result_b64 = base64.b64encode(img_bytes).decode('utf-8')
                                
                                return {
                                    "success": True,
                                    "result_image_base64": result_b64,
                                    "service_used": "fashn",
                                    "processing_time": attempt * 2
                                }
                            else:
                                raise Exception("No output image returned")
                        
                        elif status == "failed":
                            error = status_result.get("error", "Unknown error")
                            raise Exception(f"FASHN job failed: {error}")
                        
                        elif status in ["pending", "processing"]:
                            logger.info(f"[FASHN] Job {job_id} status: {status}, waiting...")
                            await asyncio.sleep(2)
                            continue
                        
                        else:
                            raise Exception(f"Unknown status: {status}")
                    
                    else:
                        raise Exception(f"Status check failed: {response.status}")
            
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                await asyncio.sleep(2)
        
        raise Exception("Job polling timeout")
    
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
    Clean external API service: FASHN → Vertex AI → Error Message
    No built-in computer vision system
    """
    
    def __init__(self):
        self.fashn = FASHNTryOnService()
        # Vertex AI will be added when available
    
    async def process_virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray,
                                  product_info: Dict) -> Dict:
        """
        Process virtual try-on with clean external API chain:
        1. FASHN (primary)
        2. Vertex AI (when available)
        3. Error message (final fallback)
        """
        
        # Try FASHN first (primary)
        if self.fashn.is_available():
            logger.info("[EXTERNAL] Trying FASHN (primary)")
            result = await self.fashn.virtual_tryon(customer_image, garment_image, product_info)
            
            if result.get("success"):
                logger.info("[EXTERNAL] ✅ FASHN succeeded")
                return result
            else:
                logger.warning(f"[EXTERNAL] FASHN failed: {result.get('error')}")
        
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