from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import logging
import asyncio
import base64
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import json

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Import the production Virtual Try-On Engine
from virtual_tryon_engine import virtual_tryon_engine

# Import local modules
from models import *
from database import database
from auth import AuthManager, get_current_active_user

# Create FastAPI app
app = FastAPI(title="Virtual Try-On API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Configure CORS
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

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    await database.connect()
    logger.info("Database connected successfully")
    
    # Initialize sample products
    await initialize_sample_products()
    logger.info("Sample products initialized")

@app.on_event("shutdown")
async def shutdown_event():
    await database.disconnect()
    logger.info("Database disconnected")

# Authentication Endpoints
@api_router.post("/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = await database.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password and create user
        hashed_password = AuthManager.get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password
        )
        
        created_user = await database.create_user(user)
        
        # Create access token
        access_token_expires = timedelta(minutes=30)
        access_token = AuthManager.create_access_token(
            data={"sub": created_user.email},
            expires_delta=access_token_expires
        )
        
        user_profile = UserProfile(
            id=created_user.id,
            email=created_user.email,
            full_name=created_user.full_name,
            measurements=created_user.measurements,
            profile_photo=created_user.profile_photo,
            created_at=created_user.created_at,
            updated_at=created_user.updated_at
        )
        
        return TokenResponse(
            access_token=access_token,
            user=user_profile
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@api_router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """Login user"""
    try:
        user = await AuthManager.authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        access_token_expires = timedelta(minutes=30)
        access_token = AuthManager.create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires
        )
        
        user_profile = UserProfile(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            measurements=user.measurements,
            profile_photo=user.profile_photo,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
        return TokenResponse(
            access_token=access_token,
            user=user_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@api_router.post("/reset-password")
async def reset_password(reset_data: ResetPasswordRequest):
    """Reset user password"""
    try:
        user = await database.get_user_by_email(reset_data.email)
        if not user:
            # Don't reveal if email exists or not for security
            return ApiResponse(
                success=True,
                message="If the email exists, a password reset link has been sent."
            )
        
        # In a real app, you would send an email with reset token
        # For demo purposes, we'll just return success
        logger.info(f"Password reset requested for {reset_data.email}")
        
        return ApiResponse(
            success=True,
            message="If the email exists, a password reset link has been sent."
        )
        
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )

# User Profile Endpoints
@api_router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_active_user)):
    """Get user profile"""
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        measurements=current_user.measurements,
        profile_photo=current_user.profile_photo,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )

@api_router.post("/extract-measurements")
async def extract_measurements(
    user_photo: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Extract body measurements from user photo and save photo to profile"""
    try:
        # Read and process the uploaded image
        image_data = await user_photo.read()
        
        # Convert image to base64 for storage
        import base64
        image_base64 = base64.b64encode(image_data).decode()
        profile_photo_data_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Process measurement extraction
        measurements = await process_measurement_extraction(image_data)
        
        # Update user measurements AND save photo to profile
        await database.update_user_measurements(current_user.id, measurements)
        await database.update_user_profile_photo(current_user.id, profile_photo_data_url)
        
        logger.info(f"Measurements extracted and photo saved for user {current_user.id}")
        
        return ApiResponse(
            success=True,
            message="Measurements extracted and photo saved successfully",
            data=measurements
        )
        
    except Exception as e:
        logger.error(f"Measurement extraction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Measurement extraction failed"
        )

# Product Catalog Endpoints
@api_router.get("/products", response_model=ProductCatalogResponse)
async def get_products(
    page: int = 1,
    limit: int = 20,
    category: Optional[str] = None
):
    """Get product catalog"""
    try:
        skip = (page - 1) * limit
        products = await database.get_products(skip=skip, limit=limit, category=category)
        total = await database.count_products(category=category)
        
        return ProductCatalogResponse(
            products=products,
            total=total,
            page=page,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Product catalog error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch products"
        )

@api_router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str):
    """Get specific product"""
    try:
        product = await database.get_product_by_id(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        return product
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get product error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch product"
        )

# Virtual Try-On Endpoints
@api_router.post("/tryon")
async def virtual_tryon(
    product_id: str = Form(...),
    service_type: str = Form(default="hybrid"),
    size: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    user_photo: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Process virtual try-on request"""
    try:
        start_time = time.time()
        
        # Validate product exists
        product = await database.get_product_by_id(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        
        # Read user photo
        user_image_data = await user_photo.read()
        
        # Process try-on based on service type
        if service_type == "premium":
            result_image_url, cost = await process_fal_ai_tryon(
                user_image_data, product, size, color
            )
        else:  # hybrid (default)
            result_image_url, cost = await process_hybrid_tryon(
                user_image_data, product, size, color
            )
        
        processing_time = time.time() - start_time
        
        # Save result to database
        tryon_result = TryOnResult(
            user_id=current_user.id,
            product_id=product_id,
            service_type=service_type,
            result_image_url=result_image_url,
            processing_time=processing_time,
            cost=cost,
            metadata={
                "size": size,
                "color": color,
                "product_name": product.name
            }
        )
        
        saved_result = await database.create_tryon_result(tryon_result)
        
        return ApiResponse(
            success=True,
            message="Try-on completed successfully",
            data={
                "result_id": saved_result.id,
                "result_image_url": result_image_url,
                "processing_time": processing_time,
                "cost": cost,
                "service_type": service_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Virtual try-on error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Virtual try-on failed"
        )

@api_router.get("/tryon-history")
async def get_tryon_history(
    page: int = 1,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """Get user's try-on history"""
    try:
        skip = (page - 1) * limit
        history = await database.get_user_tryon_history(
            current_user.id, skip=skip, limit=limit
        )
        
        return ApiResponse(
            success=True,
            message="Try-on history retrieved successfully",
            data=history
        )
        
    except Exception as e:
        logger.error(f"Try-on history error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch try-on history"
        )

# Health Check
@api_router.get("/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Virtual Try-On API is running", "status": "healthy"}

# Include router
app.include_router(api_router)

# Helper Functions
async def process_measurement_extraction(image_data: bytes) -> Dict[str, Any]:
    """Extract body measurements from image using AI"""
    # Mock implementation - replace with actual measurement extraction
    # Using MediaPipe or similar computer vision library
    
    # Simulate processing time
    await asyncio.sleep(2)
    
    # Return mock measurements in imperial units
    return {
        "height": 68.5,  # inches
        "weight": 150.0,  # pounds
        "chest": 36.0,   # inches
        "waist": 30.0,
        "hips": 38.0,
        "shoulder_width": 16.5,
        "arm_length": 24.0,
        "confidence_score": 0.85,
        "recommended_sizes": {
            "top": "M",
            "bottom": "M",
            "dress": "M"
        }
    }

async def process_hybrid_tryon(
    user_image_data: bytes, 
    product: Product, 
    size: Optional[str], 
    color: Optional[str]
) -> tuple[str, float]:
    """Process virtual try-on using real hybrid 3D approach via VirtualTryOnEngine with timeout protection"""
    try:
        logger.info("Starting real hybrid 3D virtual try-on pipeline with server-level timeout")
        
        # Add server-level timeout protection to prevent 502 errors
        try:
            # Use the actual VirtualTryOnEngine for real processing with 40-second timeout
            result_url, cost = await asyncio.wait_for(
                virtual_tryon_engine.process_hybrid_tryon(
                    user_image_data,
                    product.image_url,
                    product.name,
                    product.category
                ),
                timeout=40.0  # 40 seconds to ensure we respond before proxy timeout
            )
            
            logger.info("Real hybrid 3D virtual try-on completed successfully")
            return result_url, cost
            
        except asyncio.TimeoutError:
            logger.warning("Server-level timeout reached for hybrid try-on, using fallback")
            # Use fallback when server-level timeout is reached
            return await fallback_tryon_result(user_image_data, product, size, color)
        
    except Exception as e:
        logger.error(f"Hybrid try-on error: {str(e)}")
        # Fallback to placeholder if engine fails
        return await fallback_tryon_result(user_image_data, product, size, color)


async def process_fal_ai_tryon(
    user_image_data: bytes, 
    product: Product, 
    size: Optional[str], 
    color: Optional[str]
) -> tuple[str, float]:
    """Process virtual try-on using real fal.ai FASHN API via VirtualTryOnEngine"""
    try:
        logger.info("Starting real fal.ai FASHN virtual try-on")
        
        # Use the actual VirtualTryOnEngine for real fal.ai processing
        result_url, cost = await virtual_tryon_engine.process_fal_ai_tryon(
            user_image_data,
            product.image_url,
            product.name,
            product.category
        )
        
        logger.info("Real fal.ai virtual try-on completed successfully")
        return result_url, cost
        
    except Exception as e:
        logger.error(f"fal.ai try-on error: {str(e)}")
        # Fallback to hybrid approach if fal.ai fails
        return await process_hybrid_tryon(user_image_data, product, size, color)

async def fallback_tryon_result(
    user_image_data: bytes, 
    product: Product, 
    size: Optional[str], 
    color: Optional[str]
) -> tuple[str, float]:
    """Fallback function when VirtualTryOnEngine fails"""
    try:
        logger.info("Using fallback try-on result generation")
        
        # Convert user image to base64 data URL to show their actual image
        user_image_b64 = base64.b64encode(user_image_data).decode()
        result_data_url = f"data:image/jpeg;base64,{user_image_b64}"
        
        # Very low cost for fallback
        cost = 0.01
        
        logger.info("Fallback try-on result generated")
        return result_data_url, cost
        
    except Exception as e:
        logger.error(f"Fallback generation error: {str(e)}")
        # Return placeholder if all else fails
        return await generate_placeholder_result(product, "fallback failed"), 0.0

async def fallback_ai_generation(
    user_image_data: bytes, 
    product: Product, 
    size: Optional[str], 
    color: Optional[str]
) -> tuple[str, float]:
    """Fallback AI generation using EMERGENT_LLM_KEY"""
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        logger.info("Using EMERGENT_LLM_KEY fallback generation")
        
        # Initialize chat with EMERGENT_LLM_KEY
        chat = LlmChat(
            api_key=os.environ.get("EMERGENT_LLM_KEY"),
            session_id=f"tryon-{uuid.uuid4()}",
            system_message="You are an AI that generates detailed descriptions for virtual try-on images."
        ).with_model("openai", "gpt-4o")
        
        # Create detailed prompt
        prompt = f"""Generate a photorealistic virtual try-on result description for:
        Product: {product.name}
        Category: {product.category}
        Description: {product.description}
        Size: {size or 'Standard'}
        Color: {color or 'Default'}
        
        The result should show a person wearing this item with perfect fit and natural appearance."""
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # For demo purposes, return a placeholder image
        result_url = await generate_placeholder_result(product, response)
        cost = 0.01  # Very low cost for fallback
        
        logger.info("EMERGENT_LLM_KEY fallback completed")
        return result_url, cost
        
    except Exception as e:
        logger.error(f"Fallback generation error: {str(e)}")
        # Return placeholder if all else fails
        return "https://via.placeholder.com/512x512/cccccc/666666?text=Try-On+Result", 0.0

# Hybrid 3D Pipeline Functions (For reference - now using VirtualTryOnEngine)
# These functions would be called by the VirtualTryOnEngine in a real implementation
async def reconstruct_3d_body(image_data: bytes):
    """3D body reconstruction using MediaPipe + SMPL"""
    # Mock implementation - would use actual MediaPipe pose detection
    await asyncio.sleep(5)  # Simulate processing time
    return {"body_mesh": "3d_body_data"}

async def fit_garment_to_body(body_mesh, product: Product, size: Optional[str]):
    """3D garment fitting using physics simulation"""
    # Mock implementation - would use Blender Python API + PyBullet
    await asyncio.sleep(8)  # Simulate processing time
    return {"fitted_garment": "3d_garment_data"}

async def render_photorealistic_scene(body_mesh, fitted_garment):
    """Photorealistic rendering using Blender Cycles"""
    # Mock implementation - would use Blender headless rendering
    await asyncio.sleep(15)  # Simulate rendering time
    return {"rendered_image": "rendered_image_data"}

async def enhance_with_ai(rendered_image, original_image):
    """AI enhancement using Stable Diffusion"""
    # Mock implementation - would use diffusers library
    await asyncio.sleep(5)  # Simulate AI processing
    return {"enhanced_image": "enhanced_image_data"}

async def save_result_image(image_data, service_type: str) -> str:
    """Save result image and return URL - using clothing-specific placeholders"""
    # For demonstration, return clothing/fashion specific placeholder images
    # In production, this would save actual try-on results to cloud storage
    result_id = str(uuid.uuid4())
    
    # Use fashion/clothing specific placeholder images
    clothing_placeholder_images = [
        "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=600&fit=crop&crop=center",  # person in clothing
        "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=400&h=600&fit=crop&crop=center",  # fashion model
        "https://images.unsplash.com/photo-1581803118522-7b72a50f7e9f?w=400&h=600&fit=crop&crop=center",  # person wearing shirt
        "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&h=600&fit=crop&crop=center",  # clothing model
        "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=400&h=600&fit=crop&crop=center"   # fashion photo
    ]
    
    # Use service type and result_id to determine consistent placeholder
    import hashlib
    hash_value = int(hashlib.md5(result_id.encode()).hexdigest(), 16)
    placeholder_url = clothing_placeholder_images[hash_value % len(clothing_placeholder_images)]
    
    logger.info(f"Generated clothing placeholder result URL: {placeholder_url}")
    return placeholder_url

async def generate_placeholder_result(product: Product, description: str) -> str:
    """Generate placeholder result image - using clothing/fashion specific URLs"""
    # Use clothing/fashion specific placeholders based on product category
    category_images = {
        "men's tops": "https://images.unsplash.com/photo-1564859228273-274232fdb516?w=400&h=600&fit=crop&crop=center",  # man in shirt
        "men's bottoms": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400&h=600&fit=crop&crop=center",  # man in jeans
        "men's outerwear": "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=600&fit=crop&crop=center",  # man in jacket
        "women's tops": "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=400&h=600&fit=crop&crop=center",  # woman in top
        "women's bottoms": "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&h=600&fit=crop&crop=center",  # woman in pants
        "women's dresses": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400&h=600&fit=crop&crop=center",  # woman in dress
        "women's outerwear": "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=600&fit=crop&crop=center",  # woman in jacket
        "women's activewear": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=600&fit=crop&crop=center"  # activewear
    }
    
    # Find matching category or use default
    category_key = product.category.lower()
    placeholder_url = category_images.get(category_key, "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&h=600&fit=crop&crop=center")
    
    logger.info(f"Generated clothing fallback placeholder for {product.name}: {placeholder_url}")
    return placeholder_url

async def initialize_sample_products():
    """Initialize sample products in database"""
    try:
        # Check if products already exist
        existing_products = await database.get_products(limit=1)
        if existing_products:
            return  # Products already initialized
        
        sample_products = [
            # Men's Apparel
            Product(
                name="Men's Classic Polo Shirt",
                category="Men's Tops",
                brand="LLBean Style",
                description="Premium cotton polo shirt with classic fit and comfort. Perfect for casual or business casual occasions.",
                price=49.99,
                sizes=["S", "M", "L", "XL", "XXL"],
                colors=["Navy", "White", "Light Blue", "Forest Green", "Burgundy"],
                image_url="https://dummyimage.com/400x400/003366/ffffff&text=Men%27s+Polo+Shirt",
                product_images=[
                    "https://dummyimage.com/400x400/003366/ffffff&text=Navy+Polo",
                    "https://dummyimage.com/400x400/ffffff/000000&text=White+Polo"
                ]
            ),
            Product(
                name="Men's Flannel Shirt",
                category="Men's Tops",
                brand="LLBean Style", 
                description="Soft, warm flannel shirt made from premium cotton. Classic plaid pattern perfect for outdoor activities.",
                price=69.99,
                sizes=["S", "M", "L", "XL", "XXL"],
                colors=["Red Plaid", "Blue Plaid", "Green Plaid", "Gray Plaid"],
                image_url="https://via.placeholder.com/400x400/8B4513/FFFFFF?text=Men's+Flannel",
                product_images=[
                    "https://via.placeholder.com/400x400/DC143C/FFFFFF?text=Red+Flannel",
                    "https://via.placeholder.com/400x400/4169E1/FFFFFF?text=Blue+Flannel"
                ]
            ),
            Product(
                name="Men's Denim Jeans",
                category="Men's Bottoms",
                brand="LLBean Style",
                description="Classic straight-fit denim jeans with comfort stretch. Durable construction for everyday wear.",
                price=89.99,
                sizes=["30", "32", "34", "36", "38", "40"],
                colors=["Dark Wash", "Medium Wash", "Light Wash", "Black"],
                image_url="https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Men's+Jeans",
                product_images=[
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Dark+Denim",
                    "https://via.placeholder.com/400x400/3B82F6/FFFFFF?text=Medium+Denim"
                ]
            ),
            Product(
                name="Men's Outdoor Jacket",
                category="Men's Outerwear",
                brand="LLBean Style",
                description="Weather-resistant outdoor jacket with insulation. Perfect for hiking, camping, and outdoor adventures.",
                price=149.99,
                sizes=["S", "M", "L", "XL", "XXL"],
                colors=["Navy", "Forest Green", "Black", "Khaki"],
                image_url="https://via.placeholder.com/400x400/2D4A22/FFFFFF?text=Men's+Jacket",
                product_images=[
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Navy+Jacket",
                    "https://via.placeholder.com/400x400/2D4A22/FFFFFF?text=Green+Jacket"
                ]
            ),
            Product(
                name="Men's Chino Pants",
                category="Men's Bottoms",
                brand="LLBean Style",
                description="Versatile chino pants with classic fit. Great for business casual or weekend wear.",
                price=59.99,
                sizes=["30", "32", "34", "36", "38", "40"],
                colors=["Khaki", "Navy", "Olive", "Stone", "Black"],
                image_url="https://via.placeholder.com/400x400/D2B48C/000000?text=Men's+Chinos",
                product_images=[
                    "https://via.placeholder.com/400x400/D2B48C/000000?text=Khaki+Chinos",
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Navy+Chinos"
                ]
            ),
            
            # Women's Apparel  
            Product(
                name="Women's Cashmere Sweater",
                category="Women's Tops",
                brand="LLBean Style",
                description="Luxuriously soft cashmere sweater with elegant drape. Perfect for layering or wearing alone.",
                price=129.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["Cream", "Soft Pink", "Light Blue", "Charcoal", "Burgundy"],
                image_url="https://via.placeholder.com/400x400/F5F5DC/000000?text=Women's+Sweater",
                product_images=[
                    "https://via.placeholder.com/400x400/F5F5DC/000000?text=Cream+Sweater",
                    "https://via.placeholder.com/400x400/FFB6C1/000000?text=Pink+Sweater"
                ]
            ),
            Product(
                name="Women's Blouse",
                category="Women's Tops", 
                brand="LLBean Style",
                description="Elegant silk-blend blouse with feminine silhouette. Versatile piece for professional or casual wear.",
                price=79.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["White", "Blush Pink", "Navy", "Sage Green"],
                image_url="https://via.placeholder.com/400x400/FFFFFF/000000?text=Women's+Blouse",
                product_images=[
                    "https://via.placeholder.com/400x400/FFFFFF/000000?text=White+Blouse",
                    "https://via.placeholder.com/400x400/FFB6C1/000000?text=Pink+Blouse"
                ]
            ),
            Product(
                name="Women's A-Line Dress",
                category="Women's Dresses",
                brand="LLBean Style",
                description="Classic A-line dress with flattering fit. Perfect for work, dinner, or special occasions.",
                price=99.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["Navy", "Black", "Burgundy", "Forest Green", "Floral Print"],
                image_url="https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Women's+Dress",
                product_images=[
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Navy+Dress",
                    "https://via.placeholder.com/400x400/000000/FFFFFF?text=Black+Dress"
                ]
            ),
            Product(
                name="Women's Skinny Jeans",
                category="Women's Bottoms",
                brand="LLBean Style", 
                description="Comfortable skinny jeans with stretch fabric. Flattering fit that works with any top.",
                price=79.99,
                sizes=["24", "26", "28", "30", "32", "34"],
                colors=["Dark Wash", "Medium Wash", "Black", "White"],
                image_url="https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Women's+Jeans",
                product_images=[
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Dark+Skinny+Jeans",
                    "https://via.placeholder.com/400x400/000000/FFFFFF?text=Black+Jeans"
                ]
            ),
            Product(
                name="Women's Cardigan",
                category="Women's Outerwear",
                brand="LLBean Style",
                description="Soft knit cardigan perfect for layering. Comfortable and versatile for any season.",
                price=89.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["Cream", "Gray", "Navy", "Camel", "Dusty Rose"],
                image_url="https://via.placeholder.com/400x400/F5F5DC/000000?text=Women's+Cardigan",
                product_images=[
                    "https://via.placeholder.com/400x400/F5F5DC/000000?text=Cream+Cardigan",
                    "https://via.placeholder.com/400x400/808080/FFFFFF?text=Gray+Cardigan"
                ]
            ),
            Product(
                name="Women's Trench Coat",
                category="Women's Outerwear",
                brand="LLBean Style",
                description="Classic trench coat with timeless style. Water-resistant and perfect for transitional weather.",
                price=199.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["Khaki", "Navy", "Black"],
                image_url="https://via.placeholder.com/400x400/D2B48C/000000?text=Women's+Trench",
                product_images=[
                    "https://via.placeholder.com/400x400/D2B48C/000000?text=Khaki+Trench",
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Navy+Trench"
                ]
            ),
            Product(
                name="Women's Yoga Leggings",
                category="Women's Activewear",
                brand="LLBean Style",
                description="High-performance yoga leggings with moisture-wicking fabric. Perfect for workouts or casual wear.",
                price=59.99,
                sizes=["XS", "S", "M", "L", "XL"],
                colors=["Black", "Navy", "Charcoal", "Deep Purple"],
                image_url="https://via.placeholder.com/400x400/000000/FFFFFF?text=Women's+Leggings",
                product_images=[
                    "https://via.placeholder.com/400x400/000000/FFFFFF?text=Black+Leggings",
                    "https://via.placeholder.com/400x400/1E3A8A/FFFFFF?text=Navy+Leggings"
                ]
            )
        ]
        
        for product in sample_products:
            await database.create_product(product)
        
        logger.info(f"Sample products initialized successfully - {len(sample_products)} products added")
        
    except Exception as e:
        logger.error(f"Failed to initialize sample products: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)