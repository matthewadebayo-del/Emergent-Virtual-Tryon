from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, List
from enum import Enum

class GarmentType(str, Enum):
    AUTO = "auto"
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    SHOES = "shoes"
    OUTERWEAR = "outerwear"

class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    FAST = "fast"

class ReturnFormat(str, Enum):
    URL = "url"
    BASE64 = "base64"

class ProductInfo(BaseModel):
    name: str
    category: str
    description: Optional[str] = None
    product_id: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None

class TryOnOptions(BaseModel):
    garment_type: GarmentType = GarmentType.AUTO
    quality: QualityLevel = QualityLevel.HIGH
    return_format: ReturnFormat = ReturnFormat.URL
    preserve_background: bool = True

class TryOnRequest(BaseModel):
    customer_image_url: HttpUrl
    garment_image_url: HttpUrl
    product_info: ProductInfo
    api_key: str
    webhook_url: Optional[HttpUrl] = None
    options: TryOnOptions = TryOnOptions()
    
    # Optional pre-computed analysis data
    skip_customer_analysis: bool = False
    skip_garment_analysis: bool = False
    customer_analysis_data: Optional[Dict] = None
    garment_analysis_data: Optional[Dict] = None

class TryOnResponse(BaseModel):
    job_id: str
    status: str  # processing|completed|failed
    result_image_url: Optional[str] = None
    result_image_base64: Optional[str] = None
    processing_time: Optional[float] = None
    service_used: Optional[str] = None  # vertex_ai|fashn
    error: Optional[str] = None
    
class TryOnStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None  # 0-100
    estimated_completion: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None