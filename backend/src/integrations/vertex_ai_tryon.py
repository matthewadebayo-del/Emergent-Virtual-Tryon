import os
import base64
import logging
from typing import Dict, Optional
import numpy as np
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import json

logger = logging.getLogger(__name__)

class VertexAITryOn:
    """Google Cloud Vertex AI Virtual Try-On integration"""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.region = os.getenv("VERTEX_AI_REGION", "us-central1")
        self.endpoint_id = os.getenv("VERTEX_AI_ENDPOINT_ID")
        
        if not all([self.project_id, self.endpoint_id]):
            raise ValueError("Missing required Vertex AI configuration")
        
        # Initialize AI Platform
        aiplatform.init(project=self.project_id, location=self.region)
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{self.project_id}/locations/{self.region}/endpoints/{self.endpoint_id}"
        )
    
    async def virtual_tryon(self, customer_image: np.ndarray, garment_image: np.ndarray, 
                           customer_analysis: Dict, garment_analysis: Dict,
                           product_info: Dict) -> Dict:
        """
        Process virtual try-on using Vertex AI
        """
        try:
            logger.info("[VERTEX_AI] Starting virtual try-on processing")
            
            # Convert images to base64
            customer_b64 = self._image_to_base64(customer_image)
            garment_b64 = self._image_to_base64(garment_image)
            
            # Prepare request payload
            payload = {
                "instances": [{
                    "customer_image": customer_b64,
                    "garment_image": garment_b64,
                    "product_info": product_info,
                    "options": {
                        "preserve_background": True,
                        "quality": "high"
                    }
                }]
            }
            
            # Make prediction
            response = self.endpoint.predict(instances=payload["instances"])
            
            # Process response
            if response.predictions:
                result_image_b64 = response.predictions[0].get("result_image")
                
                return {
                    "success": True,
                    "result_image_base64": result_image_b64,
                    "service_used": "vertex_ai",
                    "processing_time": response.metadata.get("processing_time", 0)
                }
            else:
                raise Exception("No predictions returned from Vertex AI")
                
        except Exception as e:
            logger.error(f"[VERTEX_AI] Processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Vertex AI processing failed: {str(e)}",
                "service_used": "vertex_ai"
            }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        import cv2
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Convert to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def is_available(self) -> bool:
        """Check if Vertex AI service is available"""
        try:
            return bool(self.project_id and self.endpoint_id and self.endpoint)
        except:
            return False