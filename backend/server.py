from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import bcrypt
from jose import jwt, JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import base64
# from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
import asyncio
from PIL import Image
import io
import numpy as np
import tempfile
import aiofiles
import fal_client
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Firebase connection
from firebase_db import FirebaseDB

try:
    try:
        firebase_admin.initialize_app()
        firestore_client = firestore.client()
        db = FirebaseDB(firestore_client)
        print("✅ Connected to Firebase successfully using Application Default Credentials")
    except Exception as adc_error:
        print(f"Application Default Credentials failed: {adc_error}")
        
        firebase_config = {
            "type": "service_account",
            "project_id": "virtual-tryon-solution",
            "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID', ''),
            "private_key": os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
            "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL', ''),
            "client_id": os.environ.get('FIREBASE_CLIENT_ID', ''),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL', '')
        }
        
        if firebase_config.get('private_key'):
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            firestore_client = firestore.client()
            db = FirebaseDB(firestore_client)
            print("✅ Connected to Firebase successfully using service account credentials")
        else:
            print("⚠️  Firebase service account not configured, using mock database for development")
            db = FirebaseDB(None)  # Mock mode
            
except Exception as e:
    print(f"⚠️  Firebase initialization failed: {e}")
    db = FirebaseDB(None)  # Fallback to mock mode

# JWT Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# Initialize OpenAI Image Generation
llm_key = os.environ.get('EMERGENT_LLM_KEY')
# image_gen = OpenAIImageGeneration(api_key=llm_key) if llm_key else None

# Data Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    full_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    measurements: Optional[dict] = None

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
    height: float
    weight: float
    chest: float
    waist: float
    hips: float
    shoulder_width: float
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
    tryon_method: Optional[str] = "hybrid_3d"
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
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = await db.users.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)

# Authentication Routes
@api_router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name
    )
    
    await db.users.insert_one(user.dict())
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    user = await db.users.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.post("/reset-password")
async def reset_password(request: dict):
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
async def save_measurements(measurements: Measurements, current_user: User = Depends(get_current_user)):
    await db.users.update_one(
        {"id": current_user.id},
        {"$set": {"measurements": measurements.dict()}}
    )
    return {"message": "Measurements saved successfully"}

# Product Catalog Routes
@api_router.get("/products", response_model=List[Product])
async def get_products():
    # Import L.L.Bean product catalog
    from llbean_catalog import get_llbean_products
    sample_products = get_llbean_products()
    
    # Store products in database if they don't exist
    for product in sample_products:
        existing = await db.products.find_one({"name": product["name"]})
        if not existing:
            await db.products.insert_one(product)
    
    products = await db.products.find().to_list(1000)
    return [Product(**product) for product in products]

# Measurement Extraction Route
@api_router.post("/extract-measurements")
async def extract_measurements(
    user_image_base64: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Extract body measurements from user image using AI"""
    try:
        # Simulate AI measurement extraction
        # In production, this would use computer vision/AI to analyze the image
        simulated_measurements = {
            "height": round(165 + (hash(user_image_base64[:50]) % 30), 1),
            "weight": round(60 + (hash(user_image_base64[50:100]) % 25), 1),
            "chest": round(80 + (hash(user_image_base64[100:150]) % 20), 1),
            "waist": round(70 + (hash(user_image_base64[150:200]) % 20), 1),
            "hips": round(85 + (hash(user_image_base64[200:250]) % 20), 1),
            "shoulder_width": round(40 + (hash(user_image_base64[250:300]) % 10), 1)
        }
        
        # Save measurements automatically with conversion to inches
        measurements_cm = Measurements(**simulated_measurements)
        
        # Convert measurements to inches for US users
        measurements_inches = {
            "height": round(simulated_measurements["height"] / 2.54, 1),  # cm to inches
            "weight": round(simulated_measurements["weight"] * 2.205, 1),  # kg to pounds
            "chest": round(simulated_measurements["chest"] / 2.54, 1),
            "waist": round(simulated_measurements["waist"] / 2.54, 1), 
            "hips": round(simulated_measurements["hips"] / 2.54, 1),
            "shoulder_width": round(simulated_measurements["shoulder_width"] / 2.54, 1)
        }
        
        # Save measurements to backend (store in cm for consistency)
        await db.users.update_one(
            {"id": current_user.id},
            {"$set": {"measurements": measurements_cm.dict()}}
        )
        
        return {
            "measurements": measurements_inches,
            "message": "Measurements extracted and saved successfully"
        }
        
    except Exception as e:
        print(f"Error in extract_measurements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Measurement extraction failed: {str(e)}")

# Virtual Try-on Routes
@api_router.post("/tryon")
async def virtual_tryon(
    request: TryonRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        print("🚀 STARTING VIRTUAL TRY-ON PROCESS...")
        print(f"👤 User: {current_user.email}")
        print(f"📦 Product ID: {request.product_id}")
        print(f"🔧 Method: {request.tryon_method}")
        print(f"📏 Use stored measurements: {request.use_stored_measurements}")
        
        try:
            user_image_data = base64.b64decode(request.user_image_base64)
            print(f"✅ User image decoded: {len(user_image_data)} bytes")
        except Exception as e:
            print(f"❌ Failed to decode user image: {e}")
            raise HTTPException(status_code=400, detail="Invalid user image data")

        # Get measurements
        measurements = {}
        if request.use_stored_measurements:
            user_doc = await db.users.find_one({"email": current_user.email})
            if user_doc and user_doc.get("measurements"):
                measurements = user_doc["measurements"]
                print(f"📏 Using stored measurements: {measurements}")
            else:
                print("⚠️ No stored measurements found, using defaults")
                measurements = {"height": 175, "chest": 90, "waist": 80, "hips": 95}
        else:
            print("📏 Using default measurements")
            measurements = {"height": 175, "chest": 90, "waist": 80, "hips": 95}

        # Get clothing information
        clothing_description = ""
        clothing_image_data = None
        
        if request.product_id:
            print("📦 Looking up product in database...")
            product = await db.products.find_one({"id": request.product_id})
            if product:
                clothing_description = f"{product.get('name', '')} - {product.get('description', '')}"
                print(f"✅ Product found: {clothing_description}")
            else:
                print(f"⚠️ Product {request.product_id} not found in database")
                clothing_description = "Selected clothing item"
        
        if request.clothing_image_base64:
            try:
                clothing_image_data = base64.b64decode(request.clothing_image_base64)
                print(f"✅ Clothing image decoded: {len(clothing_image_data)} bytes")
                if not clothing_description:
                    clothing_description = "Uploaded clothing item"
            except Exception as e:
                print(f"❌ Failed to decode clothing image: {e}")
                raise HTTPException(status_code=400, detail="Invalid clothing image data")

        if not clothing_description:
            raise HTTPException(status_code=400, detail="No clothing item specified")

        garment_info = {
            'description': clothing_description,
            'image_base64': request.clothing_image_base64 if request.clothing_image_base64 else None,
            'category': 'clothing'
        }

        openai_key = os.environ.get('EMERGENT_LLM_KEY')
        fal_key = os.environ.get('FAL_KEY')

        from virtual_tryon_engine import process_virtual_tryon_request
        
        result = await process_virtual_tryon_request(
            user_image_base64=request.user_image_base64,
            garment_info=garment_info,
            measurements=measurements,
            method_str=request.tryon_method,
            openai_api_key=openai_key,
            fal_api_key=fal_key
        )
        
        print(f"✅ VIRTUAL TRY-ON COMPLETE!")
        print(f"👔 Clothing: {clothing_description}")
        print(f"🔧 Method: {result['technical_details']['method']}")
        print(f"📏 Recommended Size: {result['size_recommendation']}")
        
        # Save try-on result
        try:
            tryon_result = {
                "user_email": current_user.email,
                "product_id": request.product_id,
                "clothing_description": clothing_description,
                "result_image_base64": result['result_image_base64'],
                "size_recommendation": result['size_recommendation'],
                "measurements_used": measurements,
                "timestamp": datetime.utcnow(),
                "processing_method": result['technical_details']['method'],
                "tryon_method": request.tryon_method
            }
            await db.tryon_results.insert_one(tryon_result)
            print("💾 Try-on result saved to database")
        except Exception as e:
            print(f"⚠️ Failed to save try-on result: {e}")

        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"❌ Unexpected error in virtual try-on: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")

def analyze_user_image(user_image_bytes):
    """Analyze user image to extract appearance characteristics"""
    try:
        # Open and analyze the image
        image = Image.open(io.BytesIO(user_image_bytes))
        width, height = image.size
        
        # Simple analysis - in production this would use AI vision
        image_analysis = {
            "image_size": f"{width}x{height}",
            "aspect_ratio": "portrait" if height > width else "landscape" if width > height else "square",
        }
        
        # Basic characteristics that can be inferred
        characteristics = []
        
        # Infer some basic characteristics based on image properties
        if height > width:
            characteristics.append("full-body portrait orientation")
        
        return {
            "analysis": image_analysis,
            "characteristics": characteristics,
            "description_keywords": ["natural lighting", "realistic proportions", "authentic appearance"]
        }
    except Exception as e:
        print(f"Image analysis failed: {e}")
        return {
            "analysis": {"error": str(e)},
            "characteristics": ["natural appearance"],
            "description_keywords": ["realistic", "natural"]
        }

def determine_size_recommendation(measurements: dict, product_id: Optional[str] = None) -> str:
    """Enhanced size recommendation logic based on actual measurements"""
    try:
        # Convert measurements if they're in inches (assume if height > 100 it's in cm)
        height = measurements.get('height', 170)
        chest = measurements.get('chest', 90)  
        waist = measurements.get('waist', 75)
        
        # Convert to cm if measurements are in inches
        if height > 100:  # Likely in cm
            height_cm = height
            chest_cm = chest
            waist_cm = waist
        else:  # Likely in inches, convert to cm
            height_cm = height * 2.54
            chest_cm = chest * 2.54
            waist_cm = waist * 2.54
        
        print(f"Size calculation - Height: {height_cm}cm, Chest: {chest_cm}cm, Waist: {waist_cm}cm")
        
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
    try:
        results = await db.tryon_results.find({"user_id": current_user.id}).to_list(100)
        # Convert ObjectIds to strings and ensure all fields are serializable
        formatted_results = []
        for result in results:
            if '_id' in result:
                del result['_id']  # Remove MongoDB ObjectId
            formatted_results.append(result)
        return formatted_results
    except Exception as e:
        print(f"Error in get_tryon_history: {str(e)}")
        return []

# Health check
@api_router.get("/")
async def root():
    return {"message": "Virtual Try-on API is running"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Virtual Try-On API is running"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "http://localhost:3000",
        "https://project-summary-app-tunnel-1280i936.devinapps.com",
        "https://user:4e8d4e04fdb215110ed5606c09cb71b1@project-summary-app-tunnel-1280i936.devinapps.com",
        "https://project-summary-app-tunnel-4281p8o8.devinapps.com",
        "https://user:179b4381791ee74e2dcfadecefd74c7b@project-summary-app-tunnel-4281p8o8.devinapps.com",
        "https://project-summary-app-tunnel-*.devinapps.com",
        "https://user:*@project-summary-app-tunnel-*.devinapps.com"
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    if hasattr(db, 'db') and db.db:
        # Firebase client doesn't need explicit closing
        print("Firebase connection closed")
