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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Virtual Try-On API"])
security = HTTPBearer()

# Initialize services
orchestrator = VirtualTryOnOrchestrator()
customer_analyzer = CustomerImageAnalyzer()
garment_analyzer = GarmentImageAnalyzer()

# In-memory job storage (use Redis in production)
job_storage: Dict[str, Dict] = {}

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
        
        # Initialize job status
        job_storage[job_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": asyncio.get_event_loop().time()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_tryon_background, 
            job_id, 
            request.dict()
        )
        
        return TryOnResponse(
            job_id=job_id,
            status="processing"
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
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    return TryOnStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data.get("progress", 0),
        result_url=job_data.get("result_url"),
        error=job_data.get("error")
    )

async def process_tryon_background(job_id: str, request_data: Dict):
    """
    Background processing for async requests
    """
    try:
        logger.info(f"[BACKGROUND] Starting processing for job {job_id}")
        
        # Update progress
        job_storage[job_id]["progress"] = 10
        
        # Process the try-on
        result = await process_tryon_complete(request_data)
        
        # Update job with result
        if result["success"]:
            job_storage[job_id].update({
                "status": "completed",
                "progress": 100,
                "result_url": result.get("result_image_base64"),  # In production, upload to CDN
                "processing_time": result.get("processing_time"),
                "service_used": result.get("service_used")
            })
        else:
            job_storage[job_id].update({
                "status": "failed",
                "progress": 100,
                "error": result.get("error")
            })
        
        # Send webhook if provided
        webhook_url = request_data.get("webhook_url")
        if webhook_url:
            await send_webhook(webhook_url, job_storage[job_id])
            
    except Exception as e:
        logger.error(f"[BACKGROUND] Processing failed for job {job_id}: {str(e)}")
        job_storage[job_id].update({
            "status": "failed",
            "progress": 100,
            "error": str(e)
        })

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

async def send_webhook(webhook_url: str, job_data: Dict):
    """Send webhook notification"""
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=job_data, timeout=30)
    except Exception as e:
        logger.error(f"[WEBHOOK] Failed to send webhook: {str(e)}")