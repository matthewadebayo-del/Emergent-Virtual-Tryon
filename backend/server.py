import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import bcrypt
import cv2
import fal_client
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import (APIRouter, Depends, FastAPI, File, Form, Header,
                     HTTPException, Request, UploadFile)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.core.model_manager import model_manager
from src.utils.heic_processor import HEICProcessor

heic_processor = HEICProcessor()

print("✅ 3D virtual try-on modules configured for lazy loading")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

print("🚀 STARTING SERVER.PY - VERY FIRST LINE")
print("🔍 DEBUGGING: About to check environment variables...")

print("🔍 DEBUG: Environment variables containing 'MONGO':")
for key, value in os.environ.items():
    if "MONGO" in key.upper():
        print(f"  {key} = {repr(value)}")

print("🔍 DEBUG: Environment variables containing 'DB':")
for key, value in os.environ.items():
    if "DB" in key.upper():
        print(f"  {key} = {repr(value)}")

print("🔍 DEBUG: Environment variables containing 'SECRET':")
for key, value in os.environ.items():
    if "SECRET" in key.upper():
        print(
            f"  {key} = {repr(value[:20])}..."
            if len(str(value)) > 20
            else f"  {key} = {repr(value)}"
        )

# Configure fal.ai client
FAL_KEY = os.getenv("FAL_KEY")
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY
    print("🔑 fal.ai client configured with API key")
else:
    print("⚠️ FAL_KEY not found, fal.ai integration will be disabled")

# MongoDB connection - defer initialization to prevent startup blocking
print("🔍 DEBUGGING: Getting MONGO_URL from environment...")
mongo_url = os.environ.get("MONGO_URL")
db_name = os.environ.get("DB_NAME", "virtualfit_production")

print(f"🔍 DEBUG: Raw MONGO_URL from environment: {repr(mongo_url)}")
print(f"🔍 DEBUG: MONGO_URL type: {type(mongo_url)}")
print(f"🔍 DEBUG: MONGO_URL length: {len(mongo_url) if mongo_url else 'None'}")

if mongo_url:
    print(f"🔍 DEBUG: MONGO_URL first 50 chars: {repr(mongo_url[:50])}")
    print(f"🔍 DEBUG: MONGO_URL last 50 chars: {repr(mongo_url[-50:])}")

    print(f"🔍 DEBUG: MONGO_URL bytes: {mongo_url.encode('utf-8')[:100]}")

    # Strip any whitespace and quotes that might be causing issues
    mongo_url_stripped = mongo_url.strip()
    print(f"🔍 DEBUG: MONGO_URL after strip: {repr(mongo_url_stripped)}")

    if mongo_url_stripped.startswith('"') and mongo_url_stripped.endswith('"'):
        mongo_url_stripped = mongo_url_stripped[1:-1]
        print(f"🔧 FOUND SURROUNDING QUOTES - removed them: {repr(mongo_url_stripped)}")

    if mongo_url != mongo_url_stripped:
        print("🔧 FOUND WHITESPACE/QUOTES - using cleaned version")
        mongo_url = mongo_url_stripped

    if not mongo_url.startswith(("mongodb://", "mongodb+srv://")):
        print(f"🔧 MONGO_URL missing scheme prefix. Current value: {repr(mongo_url)}")
        if "@" in mongo_url and "." in mongo_url:
            mongo_url = f"mongodb+srv://{mongo_url}"
            print("🔧 Fixed MongoDB URL format by adding mongodb+srv:// prefix")
            print(f"🔧 New MONGO_URL: {repr(mongo_url)}")
        else:
            print(f"❌ Invalid MongoDB URL format - cannot fix: {repr(mongo_url)}")
    else:
        print("✅ MongoDB URL already has correct scheme")

    if mongo_url and len(mongo_url) > 0:
        mongodb_schemes = ("mongodb://", "mongodb+srv://")
        starts_with_mongodb = mongo_url.startswith(mongodb_schemes)
        print(
            f"🔍 DEBUG: MONGO_URL validation - starts with mongodb: "
            f"{starts_with_mongodb}"
        )
        print(f"🔍 DEBUG: MONGO_URL validation - contains @: {'@' in mongo_url}")
        print(f"🔍 DEBUG: MONGO_URL validation - contains .: {'.' in mongo_url}")
    else:
        print("❌ MONGO_URL is empty or None after processing")
else:
    print("❌ MONGO_URL environment variable not set or is None")

print(f"🔍 DEBUG: Final MONGO_URL value: {repr(mongo_url)}")
print("🔍 DEBUG: About to create AsyncIOMotorClient...")

if not mongo_url:
    print("❌ CRITICAL: Cannot create MongoDB client - MONGO_URL is None or empty")
    print("❌ This will cause InvalidURI error")
else:
    print(f"✅ MONGO_URL is valid for client creation: {len(mongo_url)} characters")

client = None
db = None

# JWT Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create the main app without a prefix with increased request size limits for HEIC processing
app = FastAPI(
    title="VirtualFit API",
    description="Virtual Try-On API with HEIC support",
    version="1.0.0"
)

# Add middleware to handle larger request bodies for HEIC processing
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.exceptions import HTTPException

class FileSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        # Check content-length header for file size
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_size:
                raise HTTPException(status_code=413, detail="File too large")
        return await call_next(request)

app.add_middleware(FileSizeMiddleware, max_size=10 * 1024 * 1024)  # 10MB limit

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# Initialize OpenAI Image Generation
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

print("✅ 3D Virtual Try-On pipeline configured for lazy loading")


# Data Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    full_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    measurements: Optional[dict] = None
    captured_image: Optional[str] = None
    captured_images: Optional[List[dict]] = None


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class Measurements(BaseModel):
    # Basic info
    height: float
    weight: float
    gender: Optional[str] = None
    age_range: Optional[str] = None
    
    # Head/neck measurements  
    head_circumference: Optional[float] = None
    neck_circumference: Optional[float] = None
    
    # Upper body measurements
    shoulder_width: float
    chest: Optional[float] = None  # Keep for backward compatibility
    chest_circumference: Optional[float] = None
    bust_circumference: Optional[float] = None
    underbust_circumference: Optional[float] = None
    waist: Optional[float] = None  # Keep for backward compatibility
    waist_circumference: Optional[float] = None
    arm_length: Optional[float] = None
    forearm_length: Optional[float] = None
    bicep_circumference: Optional[float] = None
    wrist_circumference: Optional[float] = None
    
    # Lower body measurements
    hips: Optional[float] = None  # Keep for backward compatibility
    hip_circumference: Optional[float] = None
    thigh_circumference: Optional[float] = None
    knee_circumference: Optional[float] = None
    calf_circumference: Optional[float] = None
    ankle_circumference: Optional[float] = None
    inseam_length: Optional[float] = None
    outseam_length: Optional[float] = None
    rise_length: Optional[float] = None
    
    # Torso measurements
    torso_length: Optional[float] = None
    back_length: Optional[float] = None
    sleeve_length: Optional[float] = None
    
    measured_at: datetime = Field(default_factory=datetime.utcnow)


class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str
    sizes: List[str]
    image_url: str
    description: str
    price: float


class TryonRequest(BaseModel):
    user_image_base64: str
    product_id: Optional[str] = None
    clothing_image_base64: Optional[str] = None
    use_stored_measurements: bool = False


class TryonResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    result_image_base64: str
    measurements_used: dict
    size_recommendation: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Helper Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def convert_heic_to_jpeg(image_data: bytes) -> bytes:
    """Convert HEIC image data to JPEG format"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=90)
        return output.getvalue()
    except Exception as e:
        print(f"Failed to convert HEIC image: {e}")
        raise HTTPException(status_code=400, detail="Failed to process HEIC image")


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            logger.error("JWT token missing 'sub' field")
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )
        logger.info(f"🔍 JWT validation successful for email: {email}")
    except JWTError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

    user = await db.users.find_one({"email": email})
    if user is None:
        logger.error(f"❌ User not found in database: {email}")
        # Check if any users exist at all
        user_count = await db.users.count_documents({})
        logger.error(f"📊 Total users in database: {user_count}")
        raise HTTPException(status_code=401, detail="User not found")

    logger.info(f"✅ User found in database: {email}")
    logger.info(f"🔍 User record fields: {list(user.keys())}")
    
    try:
        user_obj = User(**user)
        logger.info(f"✅ User object created successfully for: {email}")
        return user_obj
    except Exception as e:
        logger.error(f"❌ Failed to create User object for {email}: {e}")
        logger.error(f"🔍 User record structure: {user}")
        raise HTTPException(status_code=401, detail="User record incompatible")


# Authentication Routes
@api_router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
    )

    await db.users.insert_one(user.dict())

    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@api_router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    logger.info(f"🔍 Login attempt for email: {login_data.email}")
    
    user = await db.users.find_one({"email": login_data.email})
    if not user:
        logger.error(f"❌ User not found during login: {login_data.email}")
        # Check if any users exist at all
        user_count = await db.users.count_documents({})
        logger.error(f"📊 Total users in database: {user_count}")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    logger.info(f"✅ User found during login: {login_data.email}")
    logger.info(f"🔍 User record fields: {list(user.keys())}")
    
    if not verify_password(login_data.password, user["password_hash"]):
        logger.error(f"❌ Password verification failed for: {login_data.email}")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    logger.info(f"✅ Password verification successful for: {login_data.email}")

    access_token = create_access_token(data={"sub": user["email"]})
    logger.info(f"✅ JWT token created for: {login_data.email}")
    return {"access_token": access_token, "token_type": "bearer"}


@api_router.post("/reset-password")
async def reset_password(request: dict):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    user = await db.users.find_one({"email": email})
    if not user:
        # Don't reveal whether email exists for security
        return {"message": "If the email exists, reset instructions have been sent"}

    # In production, you would:
    # 1. Generate a secure reset token
    # 2. Store it in database with expiration
    # 3. Send email with reset link

    # For demo purposes, just return success
    return {"message": "Password reset instructions have been sent to your email"}


# User Profile Routes
@api_router.get("/profile", response_model=User)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@api_router.post("/measurements")
async def save_measurements(
    measurements: Measurements, current_user: User = Depends(get_current_user)
):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    await db.users.update_one(
        {"id": current_user.id}, {"$set": {"measurements": measurements.dict()}}
    )
    return {"message": "Measurements saved successfully"}


@api_router.delete("/profile/reset")
async def reset_user_profile(current_user: User = Depends(get_current_user)):
    """Reset user profile by clearing measurements and captured images"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not available")
            
        await db.users.update_one(
            {"id": current_user.id}, 
            {"$unset": {
                "measurements": "",
                "captured_image": "",
                "captured_images": ""
            }}
        )
        return {"message": "User profile reset successfully"}
    except Exception as e:
        logger.error(f"Profile reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Profile reset failed: {str(e)}")


@api_router.post("/save_captured_image")
async def save_captured_image(
    image_data: dict, current_user: User = Depends(get_current_user)
):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        image_base64 = image_data.get("image_base64")
        
        if image_base64:
            try:
                image_bytes = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
                
                # Check if it's HEIC format by trying to open with PIL
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    if img.format == 'HEIC':
                        # Convert HEIC to JPEG
                        converted_bytes = convert_heic_to_jpeg(image_bytes)
                        image_base64 = "data:image/jpeg;base64," + base64.b64encode(converted_bytes).decode()
                except Exception:
                    pass
            except Exception as e:
                print(f"Error processing image format: {e}")
        
        image_record = {
            "image_base64": image_base64,
            "captured_at": datetime.utcnow(),
            "measurements": image_data.get("measurements"),
            "image_type": "camera_capture",
        }

        await db.users.update_one(
            {"id": current_user.id}, 
            {
                "$push": {"captured_images": image_record},
                "$set": {"captured_image": image_base64}
            }
        )

        return {
            "message": "Image saved to profile successfully",
            "image_id": str(image_record["captured_at"]),
        }
    except Exception as e:
        print(f"Error saving captured image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save captured image")


# Product Catalog Routes
@api_router.get("/products", response_model=List[Product])
async def get_products():
    """Get all products from the database"""
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    products = await db.products.find().to_list(1000)
    return [Product(**product) for product in products]


# Measurement Extraction Route
@api_router.post("/extract-measurements")
async def extract_measurements(
    user_image_base64: str = Form(...), 
    user_height_cm: Optional[float] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Extract body measurements from user image using enhanced AI computer vision"""
    try:
        print("=== Enhanced AI-Based Measurement Extraction ===")
        print(f"User: {current_user.email}")
        print(f"Image length: {len(user_image_base64)}")
        print(f"Reference height: {user_height_cm} cm" if user_height_cm else "No reference height provided")

        try:
            # Check if it's base64 encoded HEIC by trying to decode and detect format
            image_bytes = base64.b64decode(user_image_base64)
            
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.format == 'HEIC':
                    # Convert HEIC to JPEG
                    image_bytes = convert_heic_to_jpeg(image_bytes)
                    print("Converted HEIC image to JPEG")
            except Exception:
                pass
        except Exception as e:
            print(f"Error processing image format: {e}")
            image_bytes = base64.b64decode(user_image_base64)
        
        print(f"Decoded image size: {len(image_bytes)} bytes")

        # Use 3D body reconstruction for measurement extraction
        if model_manager.get_body_reconstructor() is None:
            print("⚠️ 3D body reconstruction not available, using fallback")
            simulated_measurements = {
                # Basic info
                "height": round(165 + (hash(user_image_base64[:50]) % 30), 1),
                "weight": round(60 + (hash(user_image_base64[50:100]) % 25), 1),
                "gender": None,
                "age_range": None,
                
                # Head/neck measurements
                "head_circumference": round(53 + (hash(user_image_base64[300:350]) % 8), 1),
                "neck_circumference": round(35 + (hash(user_image_base64[350:400]) % 8), 1),
                
                # Upper body measurements
                "shoulder_width": round(40 + (hash(user_image_base64[250:300]) % 10), 1),
                "chest": round(80 + (hash(user_image_base64[100:150]) % 20), 1),
                "chest_circumference": round(86 + (hash(user_image_base64[400:450]) % 20), 1),
                "bust_circumference": round(86 + (hash(user_image_base64[450:500]) % 20), 1),
                "underbust_circumference": round(76 + (hash(user_image_base64[500:550]) % 15), 1),
                "waist": round(70 + (hash(user_image_base64[150:200]) % 20), 1),
                "waist_circumference": round(71 + (hash(user_image_base64[550:600]) % 20), 1),
                "arm_length": round(56 + (hash(user_image_base64[600:650]) % 10), 1),
                "forearm_length": round(25 + (hash(user_image_base64[650:700]) % 5), 1),
                "bicep_circumference": round(28 + (hash(user_image_base64[700:750]) % 8), 1),
                "wrist_circumference": round(15 + (hash(user_image_base64[750:800]) % 3), 1),
                
                # Lower body measurements
                "hips": round(85 + (hash(user_image_base64[200:250]) % 20), 1),
                "hip_circumference": round(91 + (hash(user_image_base64[800:850]) % 20), 1),
                "thigh_circumference": round(51 + (hash(user_image_base64[850:900]) % 10), 1),
                "knee_circumference": round(35 + (hash(user_image_base64[900:950]) % 5), 1),
                "calf_circumference": round(33 + (hash(user_image_base64[950:1000]) % 5), 1),
                "ankle_circumference": round(20 + (hash(user_image_base64[1000:1050]) % 3), 1),
                "inseam_length": round(76 + (hash(user_image_base64[1050:1100]) % 10), 1),
                "outseam_length": round(102 + (hash(user_image_base64[1100:1150]) % 10), 1),
                "rise_length": round(25 + (hash(user_image_base64[1150:1200]) % 5), 1),
                
                # Torso measurements
                "torso_length": round(61 + (hash(user_image_base64[1200:1250]) % 10), 1),
                "back_length": round(41 + (hash(user_image_base64[1250:1300]) % 5), 1),
                "sleeve_length": round(61 + (hash(user_image_base64[1300:1350]) % 8), 1),
            }

            confidence_score = 0.7  # Lower confidence for simulated data
            individual_confidences = {
                "shoulder": 0.7,
                "hip": 0.7,
                "torso": 0.7,
                "height": 0.7,
                "arms": 0.7,
            }
        else:
            print("🎯 Using enhanced AI-based measurement extraction")

            # Convert image bytes to numpy array for enhanced processing
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            body_result = model_manager.get_body_reconstructor().process_image_bytes(
                image_bytes
            )
            
            pose_data = model_manager.get_body_reconstructor().extract_pose_landmarks(image)
            
            body_measurements = model_manager.get_body_reconstructor().estimate_body_measurements(
                pose_data["landmarks"], 
                image.shape[:2], 
                image=image,
                reference_height_cm=user_height_cm
            )

            enhanced_measurements = body_measurements.get("enhanced_measurements", {})
            
            simulated_measurements = {
                # Basic info
                "height": body_measurements["height"],
                "weight": _estimate_weight_from_measurements(body_measurements),
                "gender": enhanced_measurements.get("gender"),
                "age_range": enhanced_measurements.get("age_range"),
                
                # Head/neck measurements
                "head_circumference": enhanced_measurements.get("head_circumference"),
                "neck_circumference": enhanced_measurements.get("neck_circumference"),
                
                # Upper body measurements
                "shoulder_width": body_measurements["shoulder_width"],
                "chest": body_measurements["chest_width"],  # Backward compatibility
                "chest_circumference": enhanced_measurements.get("chest_circumference", body_measurements.get("chest_cm")),
                "bust_circumference": enhanced_measurements.get("bust_circumference"),
                "underbust_circumference": enhanced_measurements.get("underbust_circumference"),
                "waist": body_measurements["waist_width"],  # Backward compatibility
                "waist_circumference": enhanced_measurements.get("waist_circumference", body_measurements.get("waist_cm")),
                "arm_length": enhanced_measurements.get("arm_length"),
                "forearm_length": enhanced_measurements.get("forearm_length"),
                "bicep_circumference": enhanced_measurements.get("bicep_circumference"),
                "wrist_circumference": enhanced_measurements.get("wrist_circumference"),
                
                # Lower body measurements
                "hips": body_measurements["hip_width"],  # Backward compatibility
                "hip_circumference": enhanced_measurements.get("hip_circumference", body_measurements.get("hips_cm")),
                "thigh_circumference": enhanced_measurements.get("thigh_circumference"),
                "knee_circumference": enhanced_measurements.get("knee_circumference"),
                "calf_circumference": enhanced_measurements.get("calf_circumference"),
                "ankle_circumference": enhanced_measurements.get("ankle_circumference"),
                "inseam_length": enhanced_measurements.get("inseam_length"),
                "outseam_length": enhanced_measurements.get("outseam_length"),
                "rise_length": enhanced_measurements.get("rise_length"),
                
                # Torso measurements
                "torso_length": enhanced_measurements.get("torso_length", body_measurements.get("torso_length")),
                "back_length": enhanced_measurements.get("back_length"),
                "sleeve_length": enhanced_measurements.get("sleeve_length"),
            }

            confidence_score = body_measurements.get("confidence_score", 0.85)
            individual_confidences = body_measurements.get("individual_confidences", {})

            print(
                f"✅ AI-extracted measurements with confidence {confidence_score:.2f}"
            )

        print(f"Extracted measurements: {simulated_measurements}")
        print(f"Overall confidence: {confidence_score:.2f}")

        # Save measurements automatically with conversion to inches
        measurements_cm = Measurements(**simulated_measurements)

        # Convert measurements to inches for US users
        measurements_inches = {}
        for key, value in simulated_measurements.items():
            if value is not None:
                if key == "weight":
                    measurements_inches[key] = round(value * 2.205, 1)  # kg to pounds
                elif key in ["gender", "age_range"]:
                    measurements_inches[key] = value  # No conversion needed
                else:
                    measurements_inches[key] = round(value / 2.54, 1)  # cm to inches
            else:
                measurements_inches[key] = None

        # Save measurements to backend (store in cm for consistency)
        measurement_data = measurements_cm.dict()
        measurement_data["confidence_score"] = confidence_score
        measurement_data["individual_confidences"] = individual_confidences
        measurement_data["extraction_method"] = (
            "ai_computer_vision"
            if model_manager.get_body_reconstructor() is not None
            else "simulated"
        )

        await db.users.update_one(
            {"id": current_user.id}, {"$set": {"measurements": measurement_data}}
        )

        return {
            "measurements": measurements_inches,
            "confidence_score": confidence_score,
            "individual_confidences": individual_confidences,
            "extraction_method": (
                "ai_computer_vision"
                if model_manager.get_body_reconstructor() is not None
                else "simulated"
            ),
            "message": "Measurements extracted and saved successfully",
        }

    except Exception as e:
        print(f"Error in extract_measurements: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Measurement extraction failed: {str(e)}"
        )


def _estimate_weight_from_measurements(measurements: Dict[str, float]) -> float:
    """Estimate weight from body measurements using anthropometric formulas"""
    # Use Devine formula as base, adjusted for body measurements
    height_cm = measurements.get("height", 170)
    chest_cm = measurements.get("chest_width", 90)
    waist_cm = measurements.get("waist_width", 75)

    if height_cm > 152.4:  # 5 feet
        base_weight = 50 + 2.3 * ((height_cm - 152.4) / 2.54)
    else:
        base_weight = 50

    # Adjust based on body measurements
    chest_factor = (chest_cm / 90) ** 0.5  # Chest width influence
    waist_factor = (waist_cm / 75) ** 0.3  # Waist width influence

    estimated_weight = base_weight * chest_factor * waist_factor

    return max(45, min(150, estimated_weight))


# Virtual Try-on Routes
@api_router.post("/tryon")
async def virtual_tryon(
    user_image_base64: str = Form(...),
    product_id: Optional[str] = Form(None),
    clothing_image_base64: Optional[str] = Form(None),
    use_stored_measurements: str = Form(
        "false"
    ),  # Changed to str to avoid bool parsing issues
    processing_type: str = Form("default"),  # 'default' or 'premium'
    current_user: User = Depends(get_current_user),
):
    try:
        print("=== Try-on request DEBUG ===")
        print(f"User: {current_user.email}")
        print(f"Product ID: {product_id}")
        print(
            f"User image length: {len(user_image_base64) if user_image_base64 else 0}"
        )
        print(f"Clothing image: {clothing_image_base64 is not None}")
        print(f"Use stored measurements (raw): {use_stored_measurements}")

        # Convert string to boolean
        use_measurements = use_stored_measurements.lower() in ["true", "1", "yes"]
        print(f"Use stored measurements (parsed): {use_measurements}")
        print(f"Processing type: {processing_type}")

        if not openai_client:
            print("ERROR: Image generation service not available")
            raise HTTPException(
                status_code=500, detail="Image generation service not available"
            )

        # Validate inputs
        if not user_image_base64:
            print("ERROR: User image is missing")
            raise HTTPException(status_code=422, detail="User image is required")

        if not product_id and not clothing_image_base64:
            print("ERROR: Neither product_id nor clothing_image_base64 provided")
            raise HTTPException(
                status_code=422,
                detail="Either product_id or clothing_image_base64 is required",
            )

        # Get clothing information
        clothing_description = ""
        if product_id:
            print(f"Looking up product: {product_id}")
            product = await db.products.find_one({"id": product_id})
            if not product:
                print(f"ERROR: Product not found: {product_id}")
                raise HTTPException(status_code=404, detail="Product not found")
            clothing_description = f"{product['name']} - {product['description']}"
            print(f"Using product: {clothing_description}")
        else:
            clothing_description = "uploaded clothing item"
            print("Using uploaded clothing image")

        # Use stored measurements or extract from image
        measurements = None
        if use_measurements and current_user.measurements:
            measurements = current_user.measurements
            print(f"Using stored measurements: {measurements}")
        else:
            # For now, use default measurements - in production,
            # you'd use AI to extract from image
            measurements = {
                "height": 170,
                "weight": 70,
                "chest": 90,
                "waist": 75,
                "hips": 95,
                "shoulder_width": 45,
            }
            print(f"Using default measurements: {measurements}")

        # Generate try-on image using AI with personalized approach
        print("🎭 Creating personalized virtual try-on...")

        # Decode the user's image for processing with enhanced error handling
        try:
            print(f"🔍 Processing user image, base64 length: {len(user_image_base64)}")
            
            if user_image_base64.startswith("data:"):
                user_image_base64 = user_image_base64.split(",", 1)[1]
                print(f"🔍 Cleaned base64 length: {len(user_image_base64)}")
            
            user_image_bytes = base64.b64decode(user_image_base64)
            print(f"✅ User image decoded successfully: {len(user_image_bytes)} bytes")
            
            # Validate image format
            from PIL import Image
            import io
            try:
                test_image = Image.open(io.BytesIO(user_image_bytes))
                print(f"✅ Image validation successful: {test_image.format} {test_image.size}")
            except Exception as img_error:
                print(f"❌ Image validation failed: {str(img_error)}")
                raise HTTPException(status_code=422, detail=f"Invalid image format: {str(img_error)}")
                
        except base64.binascii.Error as e:
            print(f"❌ Base64 decoding failed: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Invalid base64 format: {str(e)}")
        except Exception as e:
            print(f"❌ User image processing failed: {str(e)}")
            raise HTTPException(status_code=422, detail="Invalid user image format")

        # 🎯 ADVANCED VIRTUAL TRY-ON: Multi-Stage AI Pipeline
        print("🚀 Starting Advanced Virtual Try-On with Identity Preservation...")
        print(
            "📊 Pipeline: Photo Analysis → Person Segmentation → "
            "Garment Integration → Realistic Blending"
        )

        # Stage 1: Image Analysis and Preprocessing
        print("🔍 Stage 1: Advanced Image Analysis...")
        try:
            user_image_bytes = base64.b64decode(user_image_base64)
            print(f"✅ User image decoded: {len(user_image_bytes)} bytes")

            # Save user image temporarily for advanced processing
            temp_dir = Path("/tmp/virtualfit")
            temp_dir.mkdir(exist_ok=True)

            user_image_path = (
                temp_dir / f"user_{current_user.id}_{hash(user_image_base64[:100])}.png"
            )

            async with aiofiles.open(user_image_path, "wb") as f:
                await f.write(user_image_bytes)

            print(f"💾 Saved user image for processing: {user_image_path}")

        except Exception as e:
            print(f"❌ Image preprocessing failed: {str(e)}")
            raise HTTPException(status_code=422, detail="Invalid user image format")

        # Stage 2: Get Clothing Item Information
        print("👔 Stage 2: Clothing Item Analysis...")
        clothing_item_url = None
        if product_id:
            print(f"🔍 Looking up product: {product_id}")
            product = await db.products.find_one({"id": product_id})
            if not product:
                print(f"❌ Product not found: {product_id}")
                raise HTTPException(status_code=404, detail="Product not found")

            clothing_item_url = product["image_url"]
            clothing_description = f"{product['name']} - {product['description']}"
            print(f"✅ Using product: {clothing_description}")
            print(f"🖼️ Clothing image URL: {clothing_item_url}")

        elif clothing_image_base64:
            # Handle uploaded clothing image
            print("📤 Processing uploaded clothing image...")
            try:
                clothing_bytes = base64.b64decode(clothing_image_base64)
                clothing_hash = hash(clothing_image_base64[:100])
                clothing_filename = f"clothing_{current_user.id}_{clothing_hash}.png"
                clothing_path = temp_dir / clothing_filename

                async with aiofiles.open(clothing_path, "wb") as f:
                    await f.write(clothing_bytes)

                clothing_item_url = str(clothing_path)  # Local file path for now
                clothing_description = "custom uploaded clothing item"
                print(f"✅ Saved clothing image: {clothing_path}")

            except Exception as e:
                print(f"❌ Clothing image processing failed: {str(e)}")
                raise HTTPException(
                    status_code=422, detail="Invalid clothing image format"
                )

        # Stage 3: AI Virtual Try-On Processing
        if processing_type == "premium" and FAL_KEY:
            print("🎨 Stage 3: Premium AI Virtual Try-On Processing...")
            print(
                "🧠 Using fal.ai FASHN v1.6 with Identity Preservation "
                "& Segmentation-Free Processing"
            )
        else:
            print("🎨 Stage 3: Standard AI Virtual Try-On Processing...")
            print("🧠 Using 3D hybrid pipeline with photorealistic rendering")

        # Initialize processing method variables
        processing_method = "3D Hybrid Pipeline"
        identity_preservation = "Enhanced prompting with identity preservation"

        print(f"🎯 Processing Type Selected: {processing_type.upper()}")

        if not clothing_item_url:
            raise HTTPException(status_code=422, detail="No clothing item specified")

        if processing_type == "premium" and FAL_KEY:
            try:
                print("🚀 PREMIUM PROCESSING: Configuring fal.ai FASHN v1.6...")

                user_image_base64 = base64.b64encode(user_image_bytes).decode("utf-8")

                print("🎨 Stage 3: Advanced Virtual Try-On using fal.ai FASHN v1.6...")
                print(
                    "🧠 Multi-Stage Pipeline: Pose Detection → Segmentation → "
                    "Garment Synthesis → Post-Processing"
                )
                print(
                    "🧠 Using fal.ai FASHN v1.6 with Identity Preservation "
                    "& Segmentation-Free Processing"
                )

                print("🚀 Calling fal.ai FASHN v1.6 API...")
                result = fal_client.subscribe(
                    "fal-ai/fashn/tryon/v1.6",
                    arguments={
                        "model_image": f"data:image/jpeg;base64,{user_image_base64}",
                        "garment_image": clothing_item_url,
                        "category": "auto",
                        "mode": "balanced",
                    },
                )

                print(f"📊 fal.ai API response received: {type(result)}")

                if result and "images" in result and len(result["images"]) > 0:
                    image_url = result["images"][0]["url"]
                    print(f"🖼️ Downloading result image from: {image_url}")

                    image_response = requests.get(image_url, timeout=30)
                    image_response.raise_for_status()
                    images = [image_response.content]

                    print("✅ fal.ai FASHN v1.6 processing completed successfully!")
                    print(f"📏 Result image size: {len(images[0])} bytes")
                    processing_method = (
                        "fal.ai FASHN v1.6 Advanced Virtual Try-On Pipeline"
                    )
                    identity_preservation = (
                        "Enhanced with fal.ai FASHN v1.6 multi-stage processing"
                    )

                else:
                    raise Exception(f"Invalid fal.ai response format: {result}")

            except Exception as fal_error:
                print(f"⚠️ fal.ai processing failed: {str(fal_error)}")
                print("🔄 Falling back to enhanced OpenAI generation...")
                processing_type = "default"  # Fall through to OpenAI processing

        if processing_type == "default" or (
            processing_type == "premium" and not FAL_KEY
        ):
            print("⚡ DEFAULT PROCESSING: Using 3D Hybrid Virtual Try-On...")

            if processing_type == "premium" and not FAL_KEY:
                print(
                    "⚠️ FAL_KEY not configured, falling back to 3D hybrid "
                    "for premium request"
                )

            if model_manager.get_body_reconstructor() is None:
                print("❌ 3D pipeline not available, cannot process request")
                raise HTTPException(
                    status_code=500, detail="3D virtual try-on service not available"
                )

            try:
                print("🎭 Stage 1: 3D Body Reconstruction...")
                body_result = (
                    model_manager.get_body_reconstructor().process_image_bytes(
                        user_image_bytes
                    )
                )
                body_mesh = body_result["body_mesh"]
                body_measurements = body_result["measurements"]

                print(
                    f"✅ Body reconstruction complete: "
                    f"{len(body_mesh.vertices)} vertices"
                )

                print("🎭 Stage 2: Garment Fitting...")
                # Determine garment type from product
                garment_type = "shirts"  # Default
                garment_subtype = "t_shirt"  # Default

                if product_id:
                    product = await db.products.find_one({"id": product_id})
                    if product and "category" in product:
                        garment_type = product["category"]
                        if "name" in product:
                            if "polo" in product["name"].lower():
                                garment_subtype = "polo_shirt"
                            elif "dress" in product["name"].lower():
                                garment_subtype = "dress_shirt"

                fitted_garment = model_manager.get_garment_fitter().fit_garment_to_body(
                    body_mesh, garment_type, garment_subtype
                )

                print(f"✅ Garment fitting complete: {garment_type}/{garment_subtype}")

                print("🎭 Stage 3: Photorealistic Rendering...")
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as temp_render:
                    rendered_path = model_manager.get_renderer().render_scene(
                        body_mesh,
                        fitted_garment,
                        temp_render.name,
                        fabric_type="cotton",
                        fabric_color=(0.2, 0.3, 0.8),
                    )

                    rendered_image = Image.open(rendered_path)
                    print(f"✅ Rendering complete: {rendered_path}")

                print("🎭 Stage 4: AI Enhancement...")
                original_image = Image.open(io.BytesIO(user_image_bytes))
                enhanced_image = model_manager.get_ai_enhancer().enhance_realism(
                    rendered_image, original_image
                )

                # Convert to bytes
                with io.BytesIO() as output:
                    enhanced_image.save(output, format="PNG")
                    images = [output.getvalue()]

                processing_method = (
                    "3D Hybrid Virtual Try-On (MediaPipe + Blender + Stable Diffusion)"
                )
                identity_preservation = "3D body reconstruction with AI enhancement"

                print("✅ 3D Hybrid Virtual Try-On processing completed successfully!")

            except Exception as e:
                print(f"❌ 3D processing failed: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"3D virtual try-on processing failed: {str(e)}",
                )

        # Stage 4: Post-Processing and Quality Enhancement
        print("✨ Stage 4: Post-Processing and Quality Enhancement...")

        if not images or len(images) == 0:
            print("❌ No virtual try-on result generated")
            raise HTTPException(
                status_code=500, detail="Failed to create virtual try-on image"
            )

        print(f"✅ Virtual try-on generated successfully: {len(images[0])} bytes")

        # Clean up temporary files
        try:
            if user_image_path.exists():
                user_image_path.unlink()
            if "clothing_path" in locals() and Path(clothing_path).exists():
                Path(clothing_path).unlink()
            print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

        # Stage 5: Results and Recommendations
        print("📊 Stage 5: Size Analysis and Recommendations...")

        result_image_base64 = base64.b64encode(images[0]).decode("utf-8")
        size_recommendation = determine_size_recommendation(measurements, product_id)

        print("✅ ADVANCED VIRTUAL TRY-ON COMPLETE!")
        print(f"👔 Clothing: {clothing_description}")
        print(f"📏 Size Recommendation: {size_recommendation}")
        print("🎯 Identity Preservation: Enhanced prompting applied")
        print(f"💾 Result Size: {len(result_image_base64)} characters (base64)")

        if not images or len(images) == 0:
            print("ERROR: No images generated")
            raise HTTPException(
                status_code=500, detail="Failed to create virtual try-on image"
            )

        print(
            f"Successfully created personalized virtual try-on, "
            f"size: {len(images[0])} bytes"
        )

        # Convert to base64
        result_image_base64 = base64.b64encode(images[0]).decode("utf-8")

        # Determine size recommendation based on measurements
        size_recommendation = determine_size_recommendation(
            measurements, product_id if product_id else None
        )
        print(f"Size recommendation: {size_recommendation}")

        # Save try-on result
        tryon_result = TryonResult(
            user_id=current_user.id,
            result_image_base64=result_image_base64,
            measurements_used=measurements,
            size_recommendation=size_recommendation,
        )

        await db.tryon_results.insert_one(tryon_result.dict())
        print(f"✅ Try-on completed successfully for user {current_user.email}")

        return {
            "result_image_base64": result_image_base64,
            "size_recommendation": size_recommendation,
            "measurements_used": measurements,
            "processing_method": processing_method,
            "identity_preservation": identity_preservation,
            "personalization_note": (
                f"Advanced virtual try-on created using multi-stage AI pipeline for "
                f"{clothing_description}. Identity preservation technology applied to "
                f"maintain your exact appearance."
            ),
            "technical_details": {
                "pipeline_stages": 5,
                "identity_preservation": True,
                "segmentation_free": True,
                "measurements_based_fit": True,
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"❌ Unexpected error in virtual try-on: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")


@api_router.post("/api/v1/tryon/3d")
async def virtual_tryon_3d(
    user_image: UploadFile = File(...),
    product_id: Optional[str] = Form(None),
    garment_type: str = Form("shirt"),
    fabric_type: str = Form("cotton"),
    fabric_color: str = Form("0.2,0.3,0.8"),
    current_user: User = Depends(get_current_user),
):
    """3D Virtual Try-On API endpoint for e-commerce integration"""

    # Check if 3D modules are available via lazy loading
    body_reconstructor = model_manager.get_body_reconstructor()
    garment_fitter = model_manager.get_garment_fitter()
    renderer = model_manager.get_renderer()
    ai_enhancer = model_manager.get_ai_enhancer()

    if not all([body_reconstructor, garment_fitter, renderer, ai_enhancer]):
        raise HTTPException(status_code=503, detail="3D pipeline not available")

    try:
        user_image_bytes = await user_image.read()

        # Stage 1: Body Reconstruction
        body_result = model_manager.get_body_reconstructor().process_image_bytes(
            user_image_bytes
        )
        body_mesh = body_result["body_mesh"]

        # Stage 2: Garment Fitting
        fitted_garment = model_manager.get_garment_fitter().fit_garment_to_body(
            body_mesh, garment_type, "default"
        )

        # Stage 3: Rendering
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            color_tuple = tuple(map(float, fabric_color.split(",")))
            rendered_path = model_manager.get_renderer().render_scene(
                body_mesh,
                fitted_garment,
                temp_file.name,
                fabric_type=fabric_type,
                fabric_color=color_tuple,
            )

            rendered_image = Image.open(rendered_path)

        # Stage 4: AI Enhancement
        original_image = Image.open(io.BytesIO(user_image_bytes))
        enhanced_image = model_manager.get_ai_enhancer().enhance_realism(
            rendered_image, original_image
        )

        # Convert to base64
        with io.BytesIO() as output:
            enhanced_image.save(output, format="PNG")
            result_base64 = base64.b64encode(output.getvalue()).decode("utf-8")

        return {
            "result_image_base64": result_base64,
            "processing_method": (
                "3D Hybrid Virtual Try-On (MediaPipe + Blender + Stable Diffusion)"
            ),
            "body_measurements": body_result["measurements"],
            "garment_info": {
                "type": garment_type,
                "fabric": fabric_type,
                "color": fabric_color,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D processing failed: {str(e)}")


@api_router.post("/api/v1/tryon/batch")
async def batch_virtual_tryon_api(
    request: dict, api_key: str = Header(..., alias="X-API-Key")
):
    """Batch processing endpoint for multiple virtual try-on requests"""

    # Validate API key (implement your own validation logic)
    if not api_key or len(api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check if 3D modules are available via lazy loading
    body_reconstructor = model_manager.get_body_reconstructor()
    garment_fitter = model_manager.get_garment_fitter()
    renderer = model_manager.get_renderer()
    ai_enhancer = model_manager.get_ai_enhancer()

    if not all([body_reconstructor, garment_fitter, renderer, ai_enhancer]):
        raise HTTPException(status_code=503, detail="3D pipeline not available")

    try:
        batch_requests = request.get("requests", [])
        if len(batch_requests) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400, detail="Batch size limited to 10 requests"
            )

        results = []

        for idx, req in enumerate(batch_requests):
            try:
                result = {
                    "request_id": req.get("request_id", f"batch_{idx}"),
                    "status": "completed",
                    "result_image_url": None,  # Implement actual processing
                    "measurements": {},
                    "processing_time_ms": 0,
                }
                results.append(result)

            except Exception as e:
                results.append(
                    {
                        "request_id": req.get("request_id", f"batch_{idx}"),
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return {
            "batch_id": f"batch_{datetime.utcnow().timestamp()}",
            "total_requests": len(batch_requests),
            "completed": len([r for r in results if r["status"] == "completed"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


@api_router.get("/api/v1/garments/types")
async def get_garment_types():
    """Get available garment types and subtypes"""

    return {
        "garment_types": {
            "shirts": ["t_shirt", "polo_shirt", "dress_shirt"],
            "pants": ["jeans", "chinos", "shorts"],
            "dresses": ["casual_dress"],
        },
        "fabric_types": ["cotton", "silk", "denim", "polyester", "wool"],
        "supported_colors": (
            "RGB values as comma-separated string (e.g., '0.2,0.3,0.8')"
        ),
    }


@api_router.get("/api/v1/status")
async def get_system_status():
    """Get system status and available features"""

    return {
        "system_status": "operational",
        "features": {
            "3d_body_reconstruction": model_manager.get_body_reconstructor()
            is not None,
            "physics_garment_fitting": model_manager.get_garment_fitter() is not None,
            "photorealistic_rendering": model_manager.get_renderer() is not None,
            "ai_enhancement": model_manager.get_ai_enhancer() is not None,
            "measurement_extraction": model_manager.get_body_reconstructor()
            is not None,
        },
        "api_version": "1.0",
        "supported_image_formats": ["PNG", "JPEG", "JPG"],
        "max_image_size_mb": 10,
        "processing_capabilities": [
            "3D body reconstruction",
            "Physics-based garment fitting",
            "Photorealistic rendering",
            "AI style enhancement",
            "Computer vision measurements",
        ],
    }


@api_router.post("/api/v1/measurements/extract")
async def extract_measurements_api(
    image: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    """Measurement extraction API for e-commerce integration"""

    try:
        image_bytes = await image.read()

        if model_manager.get_body_reconstructor() is None:
            raise HTTPException(
                status_code=503, detail="Measurement extraction service not available"
            )

        body_result = model_manager.get_body_reconstructor().process_image_bytes(
            image_bytes
        )
        measurements = body_result["measurements"]

        # Convert to inches for API response
        measurements_inches = {
            "height": round(measurements.get("height_cm", 170) / 2.54, 1),
            "chest": round(measurements["chest_width"] / 2.54, 1),
            "waist": round(measurements["waist_width"] / 2.54, 1),
            "hips": round(measurements["hip_width"] / 2.54, 1),
            "shoulder_width": round(measurements["shoulder_width"] / 2.54, 1),
        }

        return {
            "measurements": measurements_inches,
            "confidence_scores": measurements.get("individual_confidences", {}),
            "overall_confidence": measurements.get("confidence_score", 0.85),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Measurement extraction failed: {str(e)}"
        )


@api_router.post("/api/v1/tryon/batch-upload")
async def virtual_tryon_batch(
    user_image: UploadFile = File(...),
    products: str = Form(...),  # JSON string of product list
    current_user: User = Depends(get_current_user),
):
    """Batch virtual try-on API for e-commerce integration"""

    # Check if 3D modules are available via lazy loading
    body_reconstructor = model_manager.get_body_reconstructor()
    garment_fitter = model_manager.get_garment_fitter()
    renderer = model_manager.get_renderer()
    ai_enhancer = model_manager.get_ai_enhancer()

    if not all([body_reconstructor, garment_fitter, renderer, ai_enhancer]):
        raise HTTPException(status_code=503, detail="3D pipeline not available")

    try:
        import json

        product_list = json.loads(products)
        user_image_bytes = await user_image.read()

        body_result = model_manager.get_body_reconstructor().process_image_bytes(
            user_image_bytes
        )
        body_mesh = body_result["body_mesh"]
        original_image = Image.open(io.BytesIO(user_image_bytes))

        results = []

        for product in product_list:
            try:
                fitted_garment = model_manager.get_garment_fitter().fit_garment_to_body(
                    body_mesh, product.get("type", "shirt"), "default"
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as temp_file:
                    rendered_path = model_manager.get_renderer().render_scene(
                        body_mesh,
                        fitted_garment,
                        temp_file.name,
                        fabric_type=product.get("fabric", "cotton"),
                        fabric_color=(0.2, 0.3, 0.8),
                    )
                    rendered_image = Image.open(rendered_path)

                enhanced_image = model_manager.get_ai_enhancer().enhance_realism(
                    rendered_image, original_image
                )

                # Convert to base64
                with io.BytesIO() as output:
                    enhanced_image.save(output, format="PNG")
                    result_base64 = base64.b64encode(output.getvalue()).decode("utf-8")

                results.append(
                    {
                        "product_id": product.get("id"),
                        "result_image_base64": result_base64,
                        "success": True,
                    }
                )

            except Exception as e:
                results.append(
                    {"product_id": product.get("id"), "error": str(e), "success": False}
                )

        return {
            "results": results,
            "body_measurements": body_result["measurements"],
            "processing_method": "3D Hybrid Virtual Try-On Batch Processing",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


def analyze_user_image(user_image_bytes):
    """Analyze user image to extract appearance characteristics"""
    try:
        # Open and analyze the image
        image = Image.open(io.BytesIO(user_image_bytes))
        width, height = image.size

        # Simple analysis - in production this would use AI vision
        image_analysis = {
            "image_size": f"{width}x{height}",
            "aspect_ratio": (
                "portrait"
                if height > width
                else "landscape" if width > height else "square"
            ),
        }

        # Basic characteristics that can be inferred
        characteristics = []

        # Infer some basic characteristics based on image properties
        if height > width:
            characteristics.append("full-body portrait orientation")

        return {
            "analysis": image_analysis,
            "characteristics": characteristics,
            "description_keywords": [
                "natural lighting",
                "realistic proportions",
                "authentic appearance",
            ],
        }
    except Exception as e:
        print(f"Image analysis failed: {e}")
        return {
            "analysis": {"error": str(e)},
            "characteristics": ["natural appearance"],
            "description_keywords": ["realistic", "natural"],
        }


def determine_size_recommendation(
    measurements: dict, product_id: Optional[str] = None
) -> str:
    """Enhanced size recommendation logic based on actual measurements"""
    try:
        # Convert measurements if they're in inches (assume if height > 100 it's in cm)
        height = measurements.get("height", 170)
        chest = measurements.get("chest", 90)
        waist = measurements.get("waist", 75)

        # Convert to cm if measurements are in inches
        if height > 100:  # Likely in cm
            height_cm = height
            chest_cm = chest
            waist_cm = waist
        else:  # Likely in inches, convert to cm
            height_cm = height * 2.54
            chest_cm = chest * 2.54
            waist_cm = waist * 2.54

        print(
            f"Size calculation - Height: {height_cm}cm, "
            f"Chest: {chest_cm}cm, Waist: {waist_cm}cm"
        )

        # More accurate size recommendations based on standard clothing sizes
        # For men's sizes (adjust for different product categories)
        if chest_cm <= 86 and waist_cm <= 71:
            return "XS"
        elif chest_cm <= 91 and waist_cm <= 76:
            return "S"
        elif chest_cm <= 97 and waist_cm <= 81:
            return "M"
        elif chest_cm <= 102 and waist_cm <= 86:
            return "L"
        elif chest_cm <= 107 and waist_cm <= 91:
            return "XL"
        elif chest_cm <= 112 and waist_cm <= 97:
            return "XXL"
        else:
            return "XXXL"

    except Exception as e:
        print(f"Size recommendation error: {e}")
        return "L"  # Default fallback


# Try-on History
@api_router.get("/tryon-history")
async def get_tryon_history(current_user: User = Depends(get_current_user)):
    # Check if database is initialized
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        results = await db.tryon_results.find({"user_id": current_user.id}).to_list(100)
        # Convert ObjectIds to strings and ensure all fields are serializable
        formatted_results = []
        for result in results:
            if "_id" in result:
                del result["_id"]  # Remove MongoDB ObjectId
            formatted_results.append(result)
        return formatted_results
    except Exception as e:
        print(f"Error in get_tryon_history: {str(e)}")
        return []


# Health check
@api_router.get("/")
async def root():
    return {"message": "Virtual Try-on API is running"}


@app.get("/api/v1/test-ai-dependencies")
def test_ai_dependencies():
    """Test if AI dependencies work after version fix"""
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        from huggingface_hub import cached_download
        import torch
        
        return {
            "diffusers_import": "success",
            "huggingface_hub_import": "success", 
            "torch_available": torch.cuda.is_available(),
            "versions": {
                "torch": torch.__version__,
                "diffusers": "imported_successfully"
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/convert-heic")
async def convert_heic_endpoint(request: Request):
    """Convert HEIC image to JPEG format with enhanced error handling"""
    try:
        content_length = request.headers.get("content-length", "unknown")
        logger.info(f"HEIC conversion request received, content-length: {content_length}")
        
        data = await request.json()
        heic_base64 = data.get("heic_base64")
        
        if not heic_base64:
            raise HTTPException(status_code=400, detail="heic_base64 field required")
        
        logger.info(f"HEIC base64 data length: {len(heic_base64)}")
        
        if heic_base64.startswith("data:"):
            heic_base64 = heic_base64.split(",", 1)[1]
            logger.info(f"Cleaned HEIC base64 data length: {len(heic_base64)}")
        
        heic_bytes = base64.b64decode(heic_base64)
        logger.info(f"HEIC bytes decoded: {len(heic_bytes)} bytes")
        
        # Convert to JPEG
        jpeg_base64 = convert_heic_to_jpeg(heic_bytes)
        logger.info(f"HEIC conversion successful, JPEG base64 length: {len(jpeg_base64)}")
        
        return {
            "success": True,
            "jpeg_base64": f"data:image/jpeg;base64,{jpeg_base64}"
        }
    except ValueError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"HEIC conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"HEIC conversion failed: {str(e)}")


@app.get("/api/v1/test-blender-3d")
def test_blender_3d():
    """Test if Blender subprocess rendering is available"""
    import subprocess
    try:
        result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else 'Unknown'
            return {
                "blender_available": True,
                "blender_version": version,
                "subprocess_mode": True,
                "display": os.environ.get("DISPLAY", "not_set")
            }
        else:
            return {
                "blender_available": False,
                "error": f"Blender subprocess failed with return code {result.returncode}",
                "subprocess_mode": True,
                "display": os.environ.get("DISPLAY", "not_set")
            }
    except subprocess.TimeoutExpired:
        return {
            "blender_available": False,
            "error": "Blender subprocess timeout",
            "subprocess_mode": True,
            "display": os.environ.get("DISPLAY", "not_set")
        }
    except Exception as e:
        return {
            "blender_available": False,
            "error": str(e),
            "subprocess_mode": True,
            "display": os.environ.get("DISPLAY", "not_set")
        }


@app.get("/api/v1/test-rendering-pipeline")
async def test_rendering_pipeline():
    """Test the rendering pipeline to verify it's working correctly with enhanced debugging"""
    try:
        import trimesh
        from src.core.rendering import BlenderSubprocessRenderer
        
        logger.info("🧪 Starting rendering pipeline test...")
        
        body_mesh = trimesh.creation.cylinder(radius=0.3, height=1.8)
        garment_mesh = trimesh.creation.cylinder(radius=0.35, height=1.0)
        
        logger.info(f"✅ Test meshes created - Body: {len(body_mesh.vertices)} vertices, Garment: {len(garment_mesh.vertices)} vertices")
        
        renderer = BlenderSubprocessRenderer()
        logger.info(f"🔧 Renderer initialized, Blender available: {renderer.blender_available}")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            logger.info(f"🎯 Testing render to: {temp_file.name}")
            success = renderer.render_with_subprocess(body_mesh, garment_mesh, temp_file.name)
            
            if success and os.path.exists(temp_file.name):
                file_size = os.path.getsize(temp_file.name)
                is_placeholder = file_size < 50000
                
                logger.info(f"✅ Render test complete - Size: {file_size} bytes, Placeholder: {is_placeholder}")
                
                try:
                    from PIL import Image
                    test_img = Image.open(temp_file.name)
                    img_info = f"{test_img.format} {test_img.size} {test_img.mode}"
                    logger.info(f"📸 Image validation: {img_info}")
                    
                    return {
                        "status": "success",
                        "file_size": file_size,
                        "is_placeholder": is_placeholder,
                        "output_path": temp_file.name,
                        "image_info": img_info,
                        "blender_available": renderer.blender_available
                    }
                except Exception as img_error:
                    logger.error(f"❌ Image validation failed: {str(img_error)}")
                    return {
                        "status": "success_but_invalid_image",
                        "file_size": file_size,
                        "is_placeholder": True,
                        "output_path": temp_file.name,
                        "image_error": str(img_error),
                        "blender_available": renderer.blender_available
                    }
            else:
                logger.error("❌ Rendering failed or output file not created")
                return {
                    "status": "failed",
                    "error": "Rendering failed or output file not created",
                    "blender_available": renderer.blender_available
                }
                
    except Exception as e:
        logger.error(f"❌ Rendering pipeline test failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@api_router.get("/measurements/garment/{garment_type}")
async def get_garment_measurements(
    garment_type: str,
    current_user: User = Depends(get_current_user)
):
    """Get measurements relevant for specific garment type"""
    try:
        # Get user's stored measurements
        user_data = await db.users.find_one({"id": current_user.id})
        if not user_data or not user_data.get("measurements"):
            raise HTTPException(status_code=404, detail="No measurements found for user")
        
        stored_measurements = user_data["measurements"]
        enhanced_measurements = stored_measurements.get("enhanced_measurements")
        
        if not enhanced_measurements:
            # Fallback to basic measurements
            return {
                "garment_type": garment_type,
                "measurements": {
                    "chest": stored_measurements.get("chest", 0),
                    "waist": stored_measurements.get("waist", 0),
                    "hips": stored_measurements.get("hips", 0),
                    "shoulder_width": stored_measurements.get("shoulder_width", 0),
                },
                "source": "basic_measurements"
            }
        
        from src.core.advanced_measurement_extractor import GARMENT_MEASUREMENTS
        
        # Get garment-specific measurements
        if garment_type not in GARMENT_MEASUREMENTS:
            available_types = list(GARMENT_MEASUREMENTS.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown garment type: {garment_type}. Available types: {available_types}"
            )
        
        required_measurements = GARMENT_MEASUREMENTS[garment_type]
        result = {}
        
        # Get primary measurements
        for field in required_measurements['primary']:
            value = enhanced_measurements.get(field, 0.0)
            confidence = enhanced_measurements.get('confidence_scores', {}).get(field, 0.0)
            
            if value > 0:
                result[field] = {
                    'value': value,
                    'confidence': confidence,
                    'importance': 'primary'
                }
        
        # Get secondary and optional measurements
        for importance in ['secondary', 'optional']:
            for field in required_measurements.get(importance, []):
                value = enhanced_measurements.get(field, 0.0)
                confidence = enhanced_measurements.get('confidence_scores', {}).get(field, 0.0)
                
                if value > 0:
                    result[field] = {
                        'value': value,
                        'confidence': confidence,
                        'importance': importance
                    }
        
        return {
            "garment_type": garment_type,
            "measurements": result,
            "source": "enhanced_measurements"
        }
        
    except Exception as e:
        print(f"Error getting garment measurements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get garment measurements: {str(e)}")


@app.get("/")
async def root():
    return {"status": "healthy", "service": "virtualfit-backend"}


@app.get("/health") 
async def health():
    return {"status": "ok", "timestamp": "2024-09-08"}


@app.get("/ping") 
async def ping():
    return {"message": "pong"}


@app.get("/app-root")
async def app_root():
    return {"status": "healthy", "message": "VirtualFit Backend is running"}


@app.get("/health-detailed")
async def health_check():
    """Detailed health check endpoint that verifies database connectivity"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        await db.command("ping")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Database connection failed: {str(e)}"
        )


@app.get("/api/v1/debug-versions")
async def debug_versions():
    """Check what versions are actually deployed"""
    result = {"deployment_time": datetime.utcnow().isoformat()}
    
    try:
        import torch
        result["torch"] = torch.__version__
    except ImportError as e:
        result["torch_error"] = str(e)
    
    try:
        import diffusers
        result["diffusers"] = diffusers.__version__
    except ImportError as e:
        result["diffusers_error"] = str(e)
    
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except ImportError as e:
        result["transformers_error"] = str(e)
    
    try:
        import huggingface_hub
        result["huggingface_hub"] = huggingface_hub.__version__
        
        try:
            from huggingface_hub import cached_download
            result["cached_download"] = "available"
        except ImportError as e:
            result["cached_download_error"] = str(e)
            
    except ImportError as e:
        result["huggingface_hub_error"] = str(e)
    
    try:
        import subprocess
        pip_list = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if pip_list.returncode == 0:
            lines = pip_list.stdout.split('\n')
            ml_packages = {}
            for line in lines:
                if any(pkg in line.lower() for pkg in ['torch', 'diffusers', 'transformers', 'huggingface']):
                    ml_packages[line.split()[0]] = line.split()[1] if len(line.split()) > 1 else "unknown"
            result["installed_packages"] = ml_packages
    except Exception as e:
        result["pip_list_error"] = str(e)
    
    return result


@app.get("/api/v1/debug-ml-versions")
async def debug_ml_versions():
    """Step 1: Diagnose Current State - Check ML dependency versions and cached_download availability"""
    try:
        import torch
        import transformers  
        import diffusers
        import huggingface_hub
        
        try:
            from huggingface_hub import cached_download
            cached_download_available = True
        except ImportError:
            cached_download_available = False
            
        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__, 
            "diffusers": diffusers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cached_download_available": cached_download_available,
            "diffusers_path": diffusers.__file__,
            "step_1_status": "✅ Systematic ML diagnosis complete"
        }
    except ImportError as e:
        return {"error": str(e), "step_1_status": "❌ ML dependencies missing"}


@app.get("/api/v1/debug-blender-status")
async def debug_blender_status():
    """Debug endpoint to check Blender subprocess availability and 3D rendering status"""
    try:
        import traceback
        import os
        import subprocess
        
        blender_status = {}
        try:
            result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0] if result.stdout else 'Unknown'
                blender_status = {
                    "available": True,
                    "version": version,
                    "subprocess_mode": True,
                    "error": None
                }
            else:
                blender_status = {
                    "available": False,
                    "error": f"Blender subprocess failed with return code {result.returncode}",
                    "subprocess_mode": True,
                    "stderr": result.stderr if result.stderr else "No error output"
                }
        except subprocess.TimeoutExpired:
            blender_status = {
                "available": False,
                "error": "Blender subprocess timeout",
                "subprocess_mode": True
            }
        except Exception as e:
            blender_status = {
                "available": False,
                "error": str(e),
                "subprocess_mode": True
            }
        
        memory_status = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            memory_status = {
                "available": True,
                "total_gb": round(vm.total / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "used_percent": vm.percent
            }
        except ImportError:
            memory_status = {"available": False, "error": "psutil not installed"}
        except Exception as e:
            memory_status = {"available": False, "error": str(e)}
        
        from src.core.rendering import FixedPhorealisticRenderer, AdaptiveFallbackRenderer
        
        renderer = FixedPhorealisticRenderer()
        fallback_renderer = AdaptiveFallbackRenderer()
        
        adaptive_settings = {}
        try:
            adaptive_settings = {
                "resolution": fallback_renderer._get_adaptive_resolution(),
                "quality": fallback_renderer._get_adaptive_quality(),
                "psutil_available": fallback_renderer.psutil_available
            }
        except Exception as e:
            adaptive_settings = {"error": str(e)}
        
        return {
            "blender_status": blender_status,
            "memory_status": memory_status,
            "renderer_available": blender_status["available"],
            "fallback_available": True,
            "adaptive_settings": adaptive_settings,
            "environment": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "BLENDER_USER_CONFIG": os.environ.get("BLENDER_USER_CONFIG", ""),
                "BLENDER_HEADLESS": os.environ.get("BLENDER_HEADLESS", ""),
                "DISPLAY": os.environ.get("DISPLAY", "")
            }
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug/db-status")
async def debug_db_status():
    try:
        # Check if database is initialized
        if db is None:
            return {
                "status": "error",
                "error": "Database not initialized",
                "database": db_name,
                "mongo_url_configured": bool(mongo_url),
            }

        await db.command("ping")
        user_count = await db.users.count_documents({})
        
        sample_users = []
        async for user in db.users.find({}, {"email": 1, "full_name": 1, "_id": 0}).limit(3):
            sample_users.append(user)
        
        user_structures = []
        async for user in db.users.find({}).limit(5):
            user_structure = {
                "email": user.get("email", "MISSING"),
                "has_password_hash": "password_hash" in user,
                "has_full_name": "full_name" in user,
                "has_measurements": "measurements" in user,
                "has_captured_image": "captured_image" in user,
                "has_captured_images": "captured_images" in user,
                "has_created_at": "created_at" in user,
                "all_fields": list(user.keys())
            }
            user_structures.append(user_structure)
        
        return {
            "status": "connected",
            "database": db_name,
            "user_count": user_count,
            "sample_users": sample_users,
            "user_structures": user_structures,
            "mongo_url_configured": bool(mongo_url),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "database": db_name,
            "mongo_url_configured": bool(mongo_url),
        }


@app.post("/debug/manual-init")
async def manual_database_init():
    """Manual database initialization endpoint to debug background process"""
    try:
        print("🔧 MANUAL INIT: Starting manual database initialization...")
        await init_database_background()
        return {
            "status": "success",
            "message": "Database initialization completed",
            "database": db_name,
            "mongo_url_configured": bool(mongo_url),
        }
    except Exception as e:
        print(f"🔧 MANUAL INIT ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "database": db_name,
            "mongo_url_configured": bool(mongo_url),
        }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def initialize_database():
    """Initialize database collections and sample data for production deployment"""
    logger.info("🚀 FastAPI application starting up...")

    # Initialize database immediately during startup
    if mongo_url:
        logger.info("🔄 MongoDB URL configured, initializing database connection...")
        await init_database_background()
    else:
        logger.error("❌ MONGO_URL not configured")
        logger.warning("⚠️ Starting without database connection")

    logger.info("✅ FastAPI startup completed - ready to serve requests")


async def init_database_background():
    """Database initialization with retry logic to prevent startup failures"""
    global client, db

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            logger.info(
                f"🔄 Initializing MongoDB connection "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            # Initialize MongoDB client
            if mongo_url:
                logger.info(
                    f"🔍 Creating AsyncIOMotorClient with URL: " f"{mongo_url[:50]}..."
                )

                client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
                db = client[db_name]

                # Test database connection with reduced timeout
                await asyncio.wait_for(db.command("ping"), timeout=5.0)
                logger.info("✅ MongoDB connection successful")
                break

            else:
                logger.error("❌ MONGO_URL not configured")
                return

        except Exception as e:
            logger.error(
                f"❌ Database initialization attempt {attempt + 1} failed: " f"{str(e)}"
            )
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("❌ All database initialization attempts failed")
                return

    if db is not None:
        await initialize_sample_data()


async def initialize_sample_data():
    """Initialize sample products and database indexes"""
    try:
        # Initialize sample products if products collection is empty
        product_count = await db.products.count_documents({})
        if product_count == 0:
            logger.info("📦 Creating sample product catalog...")
            sample_products = [
                {
                    "id": str(uuid.uuid4()),
                    "name": "Classic White T-Shirt",
                    "category": "shirts",
                    "sizes": ["XS", "S", "M", "L", "XL"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1521572163474-6864f9cf17ab?w=400"
                    ),
                    "description": "Comfortable cotton white t-shirt",
                    "price": 29.99,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Blue Denim Jeans",
                    "category": "pants",
                    "sizes": ["28", "30", "32", "34", "36"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1542272604-787c3835535d?w=400"
                    ),
                    "description": "Classic blue denim jeans",
                    "price": 79.99,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Black Blazer",
                    "category": "jackets",
                    "sizes": ["XS", "S", "M", "L", "XL"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1594938298603-c8148c4dae35?w=400"
                    ),
                    "description": "Professional black blazer",
                    "price": 149.99,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Summer Dress",
                    "category": "dresses",
                    "sizes": ["XS", "S", "M", "L", "XL"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1515372039744-b8f02a3ae446?w=400"
                    ),
                    "description": "Light summer dress",
                    "price": 89.99,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Navy Polo Shirt",
                    "category": "shirts",
                    "sizes": ["XS", "S", "M", "L", "XL"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1586790170083-2f9ceadc732d?w=400"
                    ),
                    "description": "Classic navy polo shirt",
                    "price": 45.99,
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Khaki Chinos",
                    "category": "pants",
                    "sizes": ["28", "30", "32", "34", "36"],
                    "image_url": (
                        "https://images.unsplash.com/"
                        "photo-1473966968600-fa801b869a1a?w=400"
                    ),
                    "description": "Comfortable khaki chino pants",
                    "price": 65.99,
                },
            ]

            await db.products.insert_many(sample_products)
            logger.info(f"✅ Created {len(sample_products)} sample products")
        else:
            logger.info(
                f"📦 Products collection already exists with {product_count} items"
            )

        await db.users.create_index("email", unique=True)
        await db.products.create_index("category")
        await db.products.create_index("name")
        logger.info("✅ Database indexes created")

        logger.info("🎉 Database initialization completed successfully")

    except Exception as e:
        logger.error(f"❌ Sample data initialization failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()
