import uuid
import logging
import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
import aiohttp
import numpy as np
from PIL import Image
import io

from ..models.api_models import TryOnRequest, TryOnResponse, TryOnStatus
from ..integrations.tryon_orchestrator import VirtualTryOnOrchestrator
from ..core.customer_image_analyzer import CustomerImageAnalyzer
from ..core.garment_analyzer import GarmentImageAnalyzer
from ..workers.tryon_tasks import process_tryon_async
from ..utils.redis_client import redis_client
from ..api.webhook_handler import send_webhook_notification, WebhookPayload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Virtual Try-On API"])
security = HTTPBearer()

# Include webhook routes
from ..api.webhook_handler import router as webhook_router
router.include_router(webhook_router, prefix="/webhook", tags=["webhooks"])

# Initialize services
orchestrator = VirtualTryOnOrchestrator()
customer_analyzer = CustomerImageAnalyzer()
garment_analyzer = GarmentImageAnalyzer()

# Redis-based job storage
# job_storage replaced with Redis client

@router.post("/tryon/process", response_model=TryOnResponse)
async def process_tryon_api(request: TryOnRequest, background_tasks: BackgroundTasks):
    """
    Main API endpoint for e-commerce integration - Async processing
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate API key (simplified - implement proper auth)
        if not await validate_api_key(request.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Start Celery task
        task = process_tryon_async.delay(
            request.dict(),
            request.webhook_url if hasattr(request, 'webhook_url') else None
        )
        
        # Store job mapping
        redis_client.set_cache(f"job:{job_id}", task.id, 3600)
        
        return TryOnResponse(
            job_id=job_id,
            status="processing",
            task_id=task.id
        )
        
    except Exception as e:
        logger.error(f"[API] Process tryon failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tryon/sync", response_model=TryOnResponse)
async def process_tryon_sync(request: TryOnRequest):
    """
    Synchronous processing for real-time needs
    """
    try:
        # Validate API key
        if not await validate_api_key(request.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Process immediately
        result = await process_tryon_complete(request.dict())
        
        return TryOnResponse(
            job_id=str(uuid.uuid4()),
            status="completed" if result["success"] else "failed",
            result_image_base64=result.get("result_image_base64"),
            processing_time=result.get("processing_time"),
            service_used=result.get("service_used"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"[API] Sync tryon failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tryon/status/{job_id}", response_model=TryOnStatus)
async def get_tryon_status(job_id: str):
    """
    Check processing status
    """
    try:
        # Get task ID from Redis
        task_id = redis_client.get_cache(f"job:{job_id}")
        if not task_id:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check Celery task status
        task = process_tryon_async.AsyncResult(task_id)
        
        status_map = {
            'PENDING': 'processing',
            'PROCESSING': 'processing', 
            'SUCCESS': 'completed',
            'FAILURE': 'failed'
        }
        
        progress = 0
        if task.state == 'PROCESSING' and task.info:
            progress = task.info.get('progress', 0)
        elif task.state == 'SUCCESS':
            progress = 100
        
        return TryOnStatus(
            job_id=job_id,
            status=status_map.get(task.state, 'unknown'),
            progress=progress,
            result_url=task.result.get('result_image_base64') if task.state == 'SUCCESS' else None,
            error=str(task.info) if task.state == 'FAILURE' else None
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tryon/result/{job_id}")
async def get_tryon_result(job_id: str):
    """
    Get completed try-on result
    """
    try:
        # Get task ID from Redis
        task_id = redis_client.get_cache(f"job:{job_id}")
        if not task_id:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check cached result first
        cached_result = redis_client.get_task_result(task_id)
        if cached_result:
            return cached_result
        
        # Check Celery task
        task = process_tryon_async.AsyncResult(task_id)
        if task.state == 'SUCCESS':
            result = task.result
            redis_client.set_task_result(task_id, result)
            return result
        elif task.state == 'FAILURE':
            raise HTTPException(status_code=400, detail=f"Processing failed: {task.info}")
        else:
            raise HTTPException(status_code=202, detail="Processing not completed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Result retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_tryon_complete(request_data: Dict) -> Dict:
    """
    Complete processing pipeline including all analysis steps
    """
    try:
        # Step 1: Download images
        customer_image = await download_image(request_data["customer_image_url"])
        garment_image = await download_image(request_data["garment_image_url"])
        
        # Step 2: Customer Analysis (unless skipped)
        if not request_data.get("skip_customer_analysis", False):
            customer_analysis = customer_analyzer.analyze_customer_image(
                image_to_bytes(customer_image)
            )
        else:
            customer_analysis = request_data.get("customer_analysis_data", {})
        
        # Step 3: Garment Analysis (unless skipped)
        if not request_data.get("skip_garment_analysis", False):
            garment_analysis = garment_analyzer.analyze_garment_image(
                image_to_bytes(garment_image)
            )
        else:
            garment_analysis = request_data.get("garment_analysis_data", {})
        
        # Step 4: Process product info
        product_info = request_data["product_info"]
        
        # Step 5: Virtual Try-On
        result = await orchestrator.process_virtual_tryon(
            customer_image, garment_image, customer_analysis,
            garment_analysis, product_info
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[COMPLETE] Processing failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to numpy array"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_bytes = await response.read()
                image = Image.open(io.BytesIO(image_bytes))
                return np.array(image)
            else:
                raise Exception(f"Failed to download image: {response.status}")

def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert numpy array to bytes"""
    pil_image = Image.fromarray(image)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

async def validate_api_key(api_key: str) -> bool:
    """Validate API key (implement proper validation)"""
    # Simplified validation - implement proper auth in production
    return len(api_key) > 10

@router.delete("/tryon/job/{job_id}")
async def cancel_tryon_job(job_id: str):
    """
    Cancel processing job
    """
    try:
        # Get task ID
        task_id = redis_client.get_cache(f"job:{job_id}")
        if not task_id:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Cancel Celery task
        task = process_tryon_async.AsyncResult(task_id)
        task.revoke(terminate=True)
        
        # Clean up Redis
        redis_client.delete_cache(f"job:{job_id}")
        redis_client.delete_cache(f"task:{task_id}")
        
        return {"message": "Job cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "service": "VirtualFit Integration API v2",
        "features": ["async_processing", "webhooks", "redis_caching"]
    }