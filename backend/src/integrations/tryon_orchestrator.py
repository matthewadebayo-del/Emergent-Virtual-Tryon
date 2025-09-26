import logging
import asyncio
from typing import Dict, Optional
import numpy as np
from .vertex_ai_tryon import VertexAITryOn
from .fashn_tryon import FASHNTryOn

logger = logging.getLogger(__name__)

class VirtualTryOnOrchestrator:
    """
    Orchestrates virtual try-on processing with primary and fallback services
    """
    
    def __init__(self):
        self.vertex_ai = VertexAITryOn() if self._vertex_ai_available() else None
        self.fashn = FASHNTryOn() if self._fashn_available() else None
        
        if not (self.vertex_ai or self.fashn):
            raise Exception("No virtual try-on services available")
    
    async def process_virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray,
                                  customer_analysis: Dict, garment_analysis: Dict,
                                  product_info: Dict) -> Dict:
        """
        Process virtual try-on with fallback mechanism
        """
        logger.info("[ORCHESTRATOR] Starting virtual try-on processing")
        
        # Try Vertex AI first (primary service)
        if self.vertex_ai and self.vertex_ai.is_available():
            try:
                logger.info("[ORCHESTRATOR] Attempting Vertex AI processing")
                result = await self.vertex_ai.virtual_tryon(
                    customer_image, garment_image, customer_analysis, 
                    garment_analysis, product_info
                )
                
                if result.get("success"):
                    logger.info("[ORCHESTRATOR] Vertex AI processing successful")
                    return result
                else:
                    logger.warning(f"[ORCHESTRATOR] Vertex AI failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Vertex AI exception: {str(e)}")
        
        # Fallback to FASHN
        if self.fashn and self.fashn.is_available():
            try:
                logger.info("[ORCHESTRATOR] Falling back to FASHN processing")
                result = await self.fashn.virtual_tryon(
                    customer_image, garment_image, customer_analysis,
                    garment_analysis, product_info
                )
                
                if result.get("success"):
                    logger.info("[ORCHESTRATOR] FASHN processing successful")
                    return result
                else:
                    logger.error(f"[ORCHESTRATOR] FASHN also failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] FASHN exception: {str(e)}")
        
        # Both services failed
        return {
            "success": False,
            "error": "All virtual try-on services unavailable",
            "service_used": "none"
        }
    
    def _vertex_ai_available(self) -> bool:
        """Check if Vertex AI is configured"""
        try:
            import os
            return bool(os.getenv("GOOGLE_CLOUD_PROJECT_ID") and os.getenv("VERTEX_AI_ENDPOINT_ID"))
        except:
            return False
    
    def _fashn_available(self) -> bool:
        """Check if FASHN is configured"""
        try:
            import os
            return bool(os.getenv("FASHN_API_KEY"))
        except:
            return False
    
    def get_available_services(self) -> Dict[str, bool]:
        """Get status of available services"""
        return {
            "vertex_ai": self.vertex_ai.is_available() if self.vertex_ai else False,
            "fashn": self.fashn.is_available() if self.fashn else False
        }