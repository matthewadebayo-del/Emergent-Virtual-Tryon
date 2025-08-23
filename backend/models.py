from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Authentication Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    full_name: str
    hashed_password: str
    is_active: bool = True
    measurements: Optional[Dict[str, Any]] = None
    profile_photo: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class UserProfile(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    measurements: Optional[Dict[str, Any]] = None
    profile_photo: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# Product Models
class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str
    brand: str
    description: str
    price: float
    sizes: List[str]
    colors: List[str]
    image_url: str
    product_images: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Virtual Try-On Models
class TryOnRequest(BaseModel):
    product_id: str
    service_type: str = "hybrid"  # "hybrid" or "premium" (fal.ai)
    size: Optional[str] = None
    color: Optional[str] = None

class TryOnResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    product_id: str
    service_type: str
    result_image_url: str
    processing_time: float
    cost: float
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MeasurementExtraction(BaseModel):
    height: float  # in inches
    weight: float  # in pounds
    chest: float
    waist: float
    hips: float
    shoulder_width: float
    arm_length: float
    confidence_score: float

# Request/Response Models
class ResetPasswordRequest(BaseModel):
    email: EmailStr

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class ProductCatalogResponse(BaseModel):
    products: List[Product]
    total: int
    page: int
    limit: int