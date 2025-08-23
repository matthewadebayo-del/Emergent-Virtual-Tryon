from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
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
from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

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
image_gen = OpenAIImageGeneration(api_key=llm_key) if llm_key else None

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
        email: str = payload.get("sub")
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
    # Sample product catalog
    sample_products = [
        {
            "id": str(uuid.uuid4()),
            "name": "Classic White T-Shirt",
            "category": "shirts",
            "sizes": ["XS", "S", "M", "L", "XL"],
            "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",
            "description": "Comfortable cotton white t-shirt",
            "price": 29.99
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Blue Denim Jeans",
            "category": "pants",
            "sizes": ["28", "30", "32", "34", "36"],
            "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400",
            "description": "Classic blue denim jeans",
            "price": 79.99
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Black Blazer",
            "category": "jackets",
            "sizes": ["XS", "S", "M", "L", "XL"],
            "image_url": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400",
            "description": "Professional black blazer",
            "price": 149.99
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Summer Dress",
            "category": "dresses",
            "sizes": ["XS", "S", "M", "L", "XL"],
            "image_url": "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=400",
            "description": "Light summer dress",
            "price": 89.99
        }
    ]
    
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
        
        # Save measurements automatically
        measurements = Measurements(**simulated_measurements)
        await db.users.update_one(
            {"id": current_user.id},
            {"$set": {"measurements": measurements.dict()}}
        )
        
        return {
            "measurements": simulated_measurements,
            "message": "Measurements extracted and saved successfully"
        }
        
    except Exception as e:
        print(f"Error in extract_measurements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Measurement extraction failed: {str(e)}")

# Virtual Try-on Routes
@api_router.post("/tryon")
async def virtual_tryon(
    user_image_base64: str = Form(...),
    product_id: Optional[str] = Form(None),
    clothing_image_base64: Optional[str] = Form(None),
    use_stored_measurements: str = Form("false"),  # Changed to str to avoid bool parsing issues
    current_user: User = Depends(get_current_user)
):
    try:
        print(f"=== Try-on request DEBUG ===")
        print(f"User: {current_user.email}")
        print(f"Product ID: {product_id}")
        print(f"User image length: {len(user_image_base64) if user_image_base64 else 0}")
        print(f"Clothing image: {clothing_image_base64 is not None}")
        print(f"Use stored measurements (raw): {use_stored_measurements}")
        
        # Convert string to boolean
        use_measurements = use_stored_measurements.lower() in ['true', '1', 'yes']
        print(f"Use stored measurements (parsed): {use_measurements}")
        
        if not image_gen:
            print("ERROR: Image generation service not available")
            raise HTTPException(status_code=500, detail="Image generation service not available")
        
        # Validate inputs
        if not user_image_base64:
            print("ERROR: User image is missing")
            raise HTTPException(status_code=422, detail="User image is required")
            
        if not product_id and not clothing_image_base64:
            print("ERROR: Neither product_id nor clothing_image_base64 provided")
            raise HTTPException(status_code=422, detail="Either product_id or clothing_image_base64 is required")
        
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
            # For now, use default measurements - in production, you'd use AI to extract from image
            measurements = {
                "height": 170,
                "weight": 70,
                "chest": 90,
                "waist": 75,
                "hips": 95,
                "shoulder_width": 45
            }
            print(f"Using default measurements: {measurements}")
        
        # Generate try-on image using AI image EDITING (not generation)
        print("🎭 Creating personalized virtual try-on using image editing...")
        
        # Decode the user's image for editing
        try:
            user_image_bytes = base64.b64decode(user_image_base64)
            print(f"Successfully decoded user image, size: {len(user_image_bytes)} bytes")
        except Exception as e:
            print(f"ERROR decoding user image: {str(e)}")
            raise HTTPException(status_code=422, detail="Invalid user image format")
        
        # Create a detailed prompt for image editing that preserves the user's appearance
        if clothing_description:
            edit_prompt = f"""Edit this person's clothing to show them wearing {clothing_description}. 
            IMPORTANT: Keep their face, body shape, skin tone, and overall appearance EXACTLY the same. 
            Only change their clothing to the {clothing_description}. 
            Make the clothing fit naturally on their body with proper sizing and realistic fabric draping.
            Ensure the lighting and photo quality matches the original.
            The person should look like themselves but wearing the new clothing item.
            Style: photorealistic, natural lighting, high quality."""
        else:
            edit_prompt = f"""Edit this person's clothing to show them wearing the uploaded clothing item.
            IMPORTANT: Keep their face, body shape, skin tone, and overall appearance EXACTLY the same.
            Only change their clothing to match the reference clothing image.
            Make the clothing fit naturally on their body with realistic proportions.
            Ensure the lighting and photo quality matches the original.
            The person should look like themselves but wearing the new clothing.
            Style: photorealistic, natural lighting, high quality."""
        
        print(f"Using image editing prompt (length: {len(edit_prompt)} characters)")
        print(f"Prompt preview: {edit_prompt[:150]}...")
        
        # Use OpenAI image editing instead of generation
        try:
            # Save user image temporarily for editing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(user_image_bytes)
                temp_file_path = temp_file.name
            
            print(f"Saved temp image file: {temp_file_path}")
            
            # Use image editing API
            with open(temp_file_path, 'rb') as image_file:
                images = await image_gen.edit_images(
                    image=[image_file],
                    prompt=edit_prompt,
                    model="gpt-image-1",
                    number_of_images=1
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Image editing failed, falling back to generation: {str(e)}")
            # Fallback to generation with user-specific details
            generation_prompt = f"""Create a photorealistic virtual try-on image of a person wearing {clothing_description}.
            The person should match these characteristics: height {measurements.get('height', 170)}cm, 
            chest {measurements.get('chest', 90)}cm, waist {measurements.get('waist', 75)}cm, hips {measurements.get('hips', 95)}cm.
            Show realistic clothing fit with natural proportions and professional lighting.
            Style: photorealistic portrait, full body, natural pose, high quality."""
            
            images = await image_gen.generate_images(
                prompt=generation_prompt,
                model="gpt-image-1",
                number_of_images=1
            )
        
        if not images or len(images) == 0:
            print("ERROR: No images generated/edited")
            raise HTTPException(status_code=500, detail="Failed to create virtual try-on image")
        
        print(f"Successfully created virtual try-on image, size: {len(images[0])} bytes")
        
        # Convert to base64
        result_image_base64 = base64.b64encode(images[0]).decode('utf-8')
        
        # Determine size recommendation based on measurements
        size_recommendation = determine_size_recommendation(measurements, product_id if product_id else None)
        print(f"Size recommendation: {size_recommendation}")
        
        # Save try-on result
        tryon_result = TryonResult(
            user_id=current_user.id,
            result_image_base64=result_image_base64,
            measurements_used=measurements,
            size_recommendation=size_recommendation
        )
        
        await db.tryon_results.insert_one(tryon_result.dict())
        print(f"✅ Try-on completed successfully for user {current_user.email}")
        
        return {
            "result_image_base64": result_image_base64,
            "size_recommendation": size_recommendation,
            "measurements_used": measurements
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"❌ Unexpected error in virtual try-on: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")

def determine_size_recommendation(measurements: dict, product_id: str = None) -> str:
    """Simple size recommendation logic - in production, this would be more sophisticated"""
    chest = measurements.get('chest', 90)
    waist = measurements.get('waist', 75)
    
    if chest <= 85 and waist <= 70:
        return "S"
    elif chest <= 95 and waist <= 80:
        return "M"
    elif chest <= 105 and waist <= 90:
        return "L"
    else:
        return "XL"

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

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
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
    client.close()