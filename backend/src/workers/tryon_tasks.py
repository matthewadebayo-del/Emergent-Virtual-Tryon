from celery import current_task
from .celery_app import celery_app
from ..integrations.tryon_orchestrator import TryOnOrchestrator
from ..models.api_models import TryOnRequest, TryOnResponse
import json
import logging
import requests

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_tryon_async(self, request_data: dict, webhook_url: str = None):
    """Process virtual try-on asynchronously"""
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # Parse request
        tryon_request = TryOnRequest(**request_data)
        
        # Initialize orchestrator
        orchestrator = TryOnOrchestrator()
        
        # Update progress
        self.update_state(state='PROCESSING', meta={'progress': 25})
        
        # Process try-on
        result = orchestrator.process_tryon(tryon_request)
        
        # Update progress
        self.update_state(state='PROCESSING', meta={'progress': 75})
        
        # Send webhook if provided
        if webhook_url and result.success:
            try:
                webhook_data = {
                    "task_id": self.request.id,
                    "status": "completed",
                    "result": result.dict()
                }
                requests.post(webhook_url, json=webhook_data, timeout=10)
            except Exception as e:
                logger.error(f"Webhook failed: {e}")
        
        # Update final state
        self.update_state(state='SUCCESS', meta={'progress': 100})
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"Async try-on failed: {e}")
        
        # Send error webhook
        if webhook_url:
            try:
                webhook_data = {
                    "task_id": self.request.id,
                    "status": "failed",
                    "error": str(e)
                }
                requests.post(webhook_url, json=webhook_data, timeout=10)
            except:
                pass
        
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise