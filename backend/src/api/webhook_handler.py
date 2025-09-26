from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

class WebhookPayload(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

class WebhookConfig(BaseModel):
    url: str
    secret: Optional[str] = None
    retry_count: int = 3

@router.post("/webhook/test")
async def test_webhook(webhook_config: WebhookConfig):
    """Test webhook endpoint connectivity"""
    try:
        test_payload = {
            "task_id": "test",
            "status": "test",
            "message": "Webhook test successful"
        }
        
        response = requests.post(
            webhook_config.url,
            json=test_payload,
            timeout=10,
            headers={"X-Webhook-Secret": webhook_config.secret} if webhook_config.secret else {}
        )
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response.text[:200]
        }
        
    except Exception as e:
        logger.error(f"Webhook test failed: {e}")
        raise HTTPException(status_code=400, detail=f"Webhook test failed: {str(e)}")

async def send_webhook_notification(webhook_url: str, payload: WebhookPayload, secret: str = None):
    """Send webhook notification with retry logic"""
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Webhook-Secret"] = secret
    
    for attempt in range(3):
        try:
            response = requests.post(
                webhook_url,
                json=payload.dict(),
                headers=headers,
                timeout=10
            )
            
            if response.status_code < 400:
                logger.info(f"Webhook sent successfully to {webhook_url}")
                return True
                
        except Exception as e:
            logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
            
    logger.error(f"All webhook attempts failed for {webhook_url}")
    return False