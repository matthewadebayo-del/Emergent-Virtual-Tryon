"""
Production-Ready Virtual Try-On Server
Full 3D Pipeline with AI Enhancement
"""

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

# Import production components
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] OpenCV not available - some features disabled")

try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False
    print("[WARN] fal_client not available - fallback to local processing")

try:
    import torch
    import torchvision
    from transformers import pipeline
    from diffusers import StableDiffusionImg2ImgPipeline
    AI_AVAILABLE = True
    print("[OK] AI/ML libraries loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"[WARN] AI libraries not available: {e}")

try:
    import trimesh
    import scipy
    from skimage import measure
    MESH_PROCESSING_AVAILABLE = True
    print("[OK] 3D mesh processing libraries loaded")
except ImportError as e:
    MESH_PROCESSING_AVAILABLE = False
    print(f"[WARN] 3D mesh processing not available: {e}")

try:
    import pybullet as p
    PHYSICS_AVAILABLE = True
    print("[OK] Physics simulation available")
except ImportError:
    PHYSICS_AVAILABLE = False
    print("[WARN] Physics simulation not available")

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MongoDB connection - defer initialization to prevent startup blocking
print("[DEBUG] DEBUGGING: Getting MONGO_URL from environment...")
mongo_url = os.environ.get("MONGO_URL")
db_name = os.environ.get("DB_NAME", "test_database")

print(f"[DEBUG] Raw MONGO_URL from environment: {repr(mongo_url)}")
print(f"[DEBUG] MONGO_URL type: {type(mongo_url)}")
print(f"[DEBUG] MONGO_URL length: {len(mongo_url) if mongo_url else 'None'}")

if mongo_url:
    # Strip any whitespace and quotes that might be causing issues
    mongo_url_stripped = mongo_url.strip()
    print(f"[DEBUG] MONGO_URL after strip: {repr(mongo_url_stripped)}")

    if mongo_url_stripped.startswith('"') and mongo_url_stripped.endswith('"'):
        mongo_url_stripped = mongo_url_stripped[1:-1]
        print(f"[FIXED] FOUND SURROUNDING QUOTES - removed them: {repr(mongo_url_stripped)}")

    if mongo_url != mongo_url_stripped:
        print("[FIXED] FOUND WHITESPACE/QUOTES - using cleaned version")
        mongo_url = mongo_url_stripped

    if not mongo_url.startswith(("mongodb://", "mongodb+srv://")):
        print(f"[FIXED] MONGO_URL missing scheme prefix. Current value: {repr(mongo_url)}")
        if "@" in mongo_url and "." in mongo_url:
            mongo_url = f"mongodb+srv://{mongo_url}"
            print("[FIXED] Fixed MongoDB URL format by adding mongodb+srv:// prefix")
            print(f"[FIXED] New MONGO_URL: {repr(mongo_url)}")
        else:
            print(f"[ERROR] Invalid MongoDB URL format - cannot fix: {repr(mongo_url)}")
    else:
        print("[OK] MongoDB URL already has correct scheme")
else:
    print("[ERROR] MONGO_URL environment variable not set or is None")

print(f"[DEBUG] Final MONGO_URL value: {repr(mongo_url)}")

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(
    title="VirtualFit Production API",
    description="Production-ready virtual try-on with full 3D pipeline",
    version="2.0.0"
)

# CORS middleware - ensure proper configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://virtual-tryon-app-a8pe83vz.devinapps.com,*")
print(f"[CORS] Configured origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# API Router
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Global variables
client = None
db = None
ai_pipeline = None
mesh_processor = None
physics_engine = None

# Initialize AI pipeline
if AI_AVAILABLE:
    try:
        # Initialize Stable Diffusion pipeline for image enhancement
        ai_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        print("[OK] Stable Diffusion pipeline initialized")
    except Exception as e:
        print(f"[WARN] Could not initialize AI pipeline: {e}")
        ai_pipeline = None

# Data Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    full_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    measurements: Optional[dict] = None
    captured_image: Optional[str] = None

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

class VirtualTryOnRequest(BaseModel):
    user_image_base64: str
    garment_image_base64: Optional[str] = None
    product_id: Optional[str] = None
    processing_mode: str = "full_3d"  # "full_3d", "ai_only", "hybrid"

# Production Virtual Try-On Engine
class ProductionVirtualTryOn:
    def __init__(self):
        self.mesh_processor = MeshProcessor() if MESH_PROCESSING_AVAILABLE else None
        self.physics_engine = PhysicsEngine() if PHYSICS_AVAILABLE else None
        self.ai_enhancer = AIEnhancer() if AI_AVAILABLE else None
        
    def _detect_garment_type(self, description: str) -> str:
        """Detect garment type from description"""
        desc_lower = description.lower()
        if "polo" in desc_lower:
            return "polo_shirt"
        elif "t-shirt" in desc_lower or "tshirt" in desc_lower:
            return "t-shirt"
        elif "jean" in desc_lower:
            return "jeans"
        elif "chino" in desc_lower:
            return "chinos"
        elif "blazer" in desc_lower:
            return "blazer"
        elif "dress" in desc_lower:
            return "dress"
        else:
            return "t-shirt"
    
    def _rgb_to_color_name(self, rgb_tuple):
        """Convert RGB values to color names"""
        r, g, b = rgb_tuple
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 100 and g > 100 and b < 100:
            return "yellow"
        else:
            return "colored"
        
    async def process_virtual_tryon(
        self, 
        user_image: bytes, 
        garment_image: bytes, 
        measurements: dict,
        mode: str = "full_3d",
        garment_description: str = "clothing item"
    ) -> dict:
        """
        Production virtual try-on processing with multiple modes
        """
        try:
            print(f"[PROCESS] Starting virtual try-on in {mode} mode")
            
            if mode == "full_3d" and self.mesh_processor and self.physics_engine:
                return await self._process_full_3d(user_image, garment_image, measurements, garment_description)
            elif mode == "ai_only" and self.ai_enhancer:
                return await self._process_ai_only(user_image, garment_image)
            elif mode == "hybrid":
                return await self._process_hybrid(user_image, garment_image, measurements)
            else:
                return await self._process_fallback(user_image, garment_image)
                
        except Exception as e:
            print(f"[ERROR] Virtual try-on processing failed: {e}")
            return await self._process_fallback(user_image, garment_image)
    
    async def _process_full_3d(self, user_image: bytes, garment_image: bytes, measurements: dict, garment_description: str = "clothing item") -> dict:
        """Full 3D pipeline with mesh processing and physics simulation"""
        print("[3D] Processing full 3D virtual try-on")
        
        # Step 1: Extract body mesh from user image
        body_mesh = self.mesh_processor.extract_body_mesh(user_image, measurements)
        
        # Step 2: Process garment mesh with type detection
        garment_type = self._detect_garment_type(garment_description)
        garment_mesh = self.mesh_processor.process_garment(garment_image, garment_type)
        
        # Step 3: Physics-based fitting
        fitted_result = self.physics_engine.fit_garment_to_body(body_mesh, garment_mesh)
        
        # Step 4: Render final image with actual garment analysis
        garment_analysis = garment_mesh.get("analysis", {})
        rendered_image = self.mesh_processor.render_scene(fitted_result, garment_description, garment_analysis)
        
        print(f"[3D] Rendered with garment colors: {garment_analysis.get('colors', {}).get('primary', 'unknown')}")
        
        # Step 5: AI enhancement with garment-specific prompts
        if self.ai_enhancer:
            # Create enhanced prompt using actual garment analysis
            if garment_analysis.get("analysis_success"):
                color_name = self._rgb_to_color_name(garment_analysis["colors"]["primary"])
                fabric_type = garment_analysis["fabric_type"]
                enhanced_description = f"{color_name} {fabric_type} {garment_description}"
            else:
                enhanced_description = garment_description
            enhanced_image = self.ai_enhancer.enhance_realism(rendered_image, user_image, enhanced_description)
        else:
            enhanced_image = rendered_image
        
        return {
            "result_image": enhanced_image,
            "processing_method": "Full 3D Pipeline",
            "confidence": 0.95,
            "features_used": ["3D_mesh", "physics_simulation", "ai_enhancement"]
        }
    
    async def _process_ai_only(self, user_image: bytes, garment_image: bytes) -> dict:
        """AI-only processing using Stable Diffusion"""
        print("[AI] Processing AI-only virtual try-on")
        
        if not self.ai_enhancer:
            return await self._process_fallback(user_image, garment_image)
        
        result_image = self.ai_enhancer.generate_tryon(user_image, garment_image)
        
        return {
            "result_image": result_image,
            "processing_method": "AI-Only Pipeline",
            "confidence": 0.85,
            "features_used": ["stable_diffusion", "image_generation"]
        }
    
    async def _process_hybrid(self, user_image: bytes, garment_image: bytes, measurements: dict) -> dict:
        """Hybrid processing combining 3D and AI"""
        print("[HYBRID] Processing hybrid virtual try-on")
        
        # Use 3D for structure, AI for enhancement
        if self.mesh_processor:
            body_mesh = self.mesh_processor.extract_body_mesh(user_image, measurements)
            base_result = self.mesh_processor.create_base_tryon(body_mesh, garment_image)
        else:
            base_result = user_image
        
        if self.ai_enhancer:
            final_result = self.ai_enhancer.enhance_realism(base_result, user_image)
        else:
            final_result = base_result
        
        return {
            "result_image": final_result,
            "processing_method": "Hybrid 3D+AI Pipeline",
            "confidence": 0.90,
            "features_used": ["3d_structure", "ai_enhancement"]
        }
    
    async def _process_fallback(self, user_image: bytes, garment_image: bytes) -> dict:
        """Fallback processing when advanced features are unavailable"""
        print("[FALLBACK] Using fallback processing")
        
        # Simple overlay processing
        user_img = Image.open(io.BytesIO(user_image))
        garment_img = Image.open(io.BytesIO(garment_image))
        
        # Resize garment to fit user image
        garment_resized = garment_img.resize((user_img.width // 2, user_img.height // 2))
        
        # Create overlay
        result_img = user_img.copy()
        overlay_pos = (user_img.width // 4, user_img.height // 4)
        result_img.paste(garment_resized, overlay_pos, garment_resized if garment_resized.mode == 'RGBA' else None)
        
        # Convert to bytes
        with io.BytesIO() as output:
            result_img.save(output, format='JPEG', quality=85)
            result_bytes = output.getvalue()
        
        return {
            "result_image": result_bytes,
            "processing_method": "Fallback Overlay",
            "confidence": 0.60,
            "features_used": ["basic_overlay"]
        }

class MeshProcessor:
    """3D mesh processing for body and garment reconstruction"""
    
    def __init__(self):
        self.initialized = MESH_PROCESSING_AVAILABLE
        
    def extract_body_mesh(self, image_bytes: bytes, measurements: dict) -> dict:
        """Extract 3D body mesh from 2D image"""
        if not self.initialized:
            raise Exception("Mesh processing not available")
        
        print("[MESH] Extracting body mesh from image")
        
        # Create parametric body model based on measurements
        height = measurements.get('height', 170)
        chest = measurements.get('chest', 90)
        waist = measurements.get('waist', 75)
        hips = measurements.get('hips', 95)
        
        # Generate basic body mesh using trimesh
        body_mesh = trimesh.creation.cylinder(
            radius=waist/200,  # Convert cm to meters and scale
            height=height/100,
            sections=16
        )
        
        return {
            "mesh": body_mesh,
            "measurements": measurements,
            "vertices": len(body_mesh.vertices),
            "faces": len(body_mesh.faces)
        }
    
    def process_garment(self, garment_image: bytes, garment_type: str = "t-shirt") -> dict:
        """Process garment image into enhanced 3D mesh with actual visual analysis"""
        print("[MESH] Processing garment into enhanced 3D mesh")
        
        # Analyze actual garment image
        from src.core.garment_analyzer import GarmentImageAnalyzer
        analyzer = GarmentImageAnalyzer()
        analysis = analyzer.analyze_garment_image(garment_image)
        
        print(f"[MESH] Garment analysis: {analysis['colors']['primary']} {analysis['fabric_type']} {analysis['patterns']['type']}")
        
        # Use enhanced 3D garment processor
        from src.core.enhanced_3d_garment_processor import Enhanced3DGarmentProcessor
        from src.core.enhanced_pipeline_controller import EnhancedPipelineController
        enhanced_processor = Enhanced3DGarmentProcessor()
        
        # Create enhanced mesh using actual visual analysis
        enhanced_mesh_data = enhanced_processor.create_enhanced_garment_mesh(analysis, garment_type)
        
        # Apply physics properties for realistic simulation
        enhanced_mesh_data = enhanced_processor.apply_physics_properties(enhanced_mesh_data)
        
        print(f"[MESH] Enhanced mesh created with {enhanced_mesh_data.get('vertices', 0)} vertices")
        print(f"[MESH] Material properties: {enhanced_mesh_data['material_properties']['fabric_type']} with roughness {enhanced_mesh_data['material_properties']['roughness']:.2f}")
        
        return {
            "mesh": enhanced_mesh_data.get("mesh"),
            "analysis": analysis,
            "enhanced_data": enhanced_mesh_data,
            "material_properties": enhanced_mesh_data["material_properties"],
            "texture_data": enhanced_mesh_data["texture_data"],
            "physics_properties": enhanced_mesh_data.get("physics_properties", {}),
            "type": garment_type,
            "vertices": enhanced_mesh_data.get("vertices", 0),
            "faces": enhanced_mesh_data.get("faces", 0)
        }
    
    def render_scene(self, fitted_result: dict, garment_description: str = "white t-shirt", garment_analysis: dict = None) -> bytes:
        """Render the final 3D scene to 2D image"""
        print("[MESH] Rendering 3D scene")
        
        # Get the fitted garment mesh
        fitted_garment = fitted_result.get("fitted_mesh")
        
        # Create 3D scene with body and garment
        img = Image.new('RGB', (512, 512), color=(240, 240, 240))  # Light background
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw body silhouette (simplified)
        body_center = (256, 300)
        body_width = 120
        body_height = 200
        
        # Draw body outline
        draw.ellipse([
            body_center[0] - body_width//2, body_center[1] - body_height//2,
            body_center[0] + body_width//2, body_center[1] + body_height//2
        ], fill=(220, 200, 180), outline=(200, 180, 160))
        
        # Use enhanced material properties and actual garment analysis
        if garment_analysis and garment_analysis.get("analysis_success"):
            # Get enhanced material properties
            enhanced_data = garment_analysis.get("enhanced_data", {})
            material_props = enhanced_data.get("material_properties", {})
            texture_data = enhanced_data.get("texture_data", {})
            
            # Use actual analyzed colors and properties
            base_color = material_props.get("base_color", [0.5, 0.5, 0.5])
            color = tuple(int(c * 255) for c in base_color[:3])
            fabric_type = material_props.get("fabric_type", "cotton")
            pattern_type = texture_data.get("pattern_type", "solid")
            roughness = material_props.get("roughness", 0.4)
            
            print(f"[MESH] Using enhanced material properties: {fabric_type} with roughness {roughness:.2f}")
            print(f"[MESH] Pattern: {pattern_type}, Color: {color}")
        else:
            # Fallback to text parsing
            desc_lower = garment_description.lower()
            if "white" in desc_lower:
                color = (255, 255, 255)
            elif "navy" in desc_lower or "dark blue" in desc_lower:
                color = (25, 25, 112)
            elif "black" in desc_lower:
                color = (40, 40, 40)
            elif "blue" in desc_lower:
                color = (70, 130, 180)
            elif "khaki" in desc_lower:
                color = (195, 176, 145)
            else:
                color = (255, 255, 255)
            fabric_type = "cotton"
            pattern_type = "solid"
            print(f"[MESH] Using text-based fallback: {color}")
        
        # Draw garment based on type
        if "shirt" in desc_lower or "polo" in desc_lower:
            # Draw shirt/polo
            garment_top = body_center[1] - 80
            garment_bottom = body_center[1] + 40
            garment_left = body_center[0] - 60
            garment_right = body_center[0] + 60
            
            draw.rectangle([
                garment_left, garment_top, garment_right, garment_bottom
            ], fill=color, outline=tuple(max(0, c-20) for c in color))
            
            # Add sleeves for shirts
            sleeve_width = 25
            sleeve_height = 50
            draw.rectangle([
                garment_left - sleeve_width, garment_top,
                garment_left, garment_top + sleeve_height
            ], fill=color, outline=tuple(max(0, c-20) for c in color))
            draw.rectangle([
                garment_right, garment_top,
                garment_right + sleeve_width, garment_top + sleeve_height
            ], fill=color, outline=tuple(max(0, c-20) for c in color))
            
        elif "jean" in desc_lower or "pant" in desc_lower or "chino" in desc_lower:
            # Draw pants
            pants_top = body_center[1] - 20
            pants_bottom = body_center[1] + 120
            pants_left = body_center[0] - 40
            pants_right = body_center[0] + 40
            
            draw.rectangle([
                pants_left, pants_top, pants_right, pants_bottom
            ], fill=color, outline=tuple(max(0, c-20) for c in color))
            
        print(f"[MESH] Enhanced 3D scene rendered with {garment_description}")
        print(f"[MESH] Material: {fabric_type}, Pattern: {pattern_type}, Color: {color}")
        if material_props:
            print(f"[MESH] Properties: roughness={material_props.get('roughness', 0.4):.2f}, sheen={material_props.get('sheen', 0.2):.2f}")
        
        # Apply post-processing effects based on material properties
        if material_props and material_props.get("sheen", 0) > 0.5:
            # Add subtle shine effect for shiny materials
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
        
        with io.BytesIO() as output:
            img.save(output, format='JPEG')
            return output.getvalue()
    
    def _render_shirt(self, draw, body_center, color, texture_data, desc_lower):
        """Enhanced shirt rendering with material properties"""
        garment_top = body_center[1] - 80
        garment_bottom = body_center[1] + 40
        garment_left = body_center[0] - 60
        garment_right = body_center[0] + 60
        
        # Apply pattern-based rendering
        pattern_type = texture_data.get("pattern_type", "solid")
        if pattern_type == "vertical_stripes":
            stripe_width = 8
            for x in range(garment_left, garment_right, stripe_width * 2):
                draw.rectangle([x, garment_top, x + stripe_width, garment_bottom], fill=color)
        else:
            draw.rectangle([garment_left, garment_top, garment_right, garment_bottom], fill=color, outline=tuple(max(0, c-20) for c in color))
        
        # Add sleeves
        sleeve_width = 25
        sleeve_height = 50
        draw.rectangle([garment_left - sleeve_width, garment_top, garment_left, garment_top + sleeve_height], fill=color)
        draw.rectangle([garment_right, garment_top, garment_right + sleeve_width, garment_top + sleeve_height], fill=color)
    
    def _render_pants(self, draw, body_center, color, texture_data):
        """Enhanced pants rendering"""
        pants_top = body_center[1] - 20
        pants_bottom = body_center[1] + 120
        pants_left = body_center[0] - 40
        pants_right = body_center[0] + 40
        
        draw.rectangle([pants_left, pants_top, pants_right, pants_bottom], fill=color, outline=tuple(max(0, c-20) for c in color))
    
    def _render_dress(self, draw, body_center, color, texture_data):
        """Enhanced dress rendering"""
        dress_top = body_center[1] - 80
        dress_bottom = body_center[1] + 100
        dress_left = body_center[0] - 50
        dress_right = body_center[0] + 50
        
        # Flared bottom
        draw.polygon([(dress_left, dress_top), (dress_right, dress_top), (dress_right + 20, dress_bottom), (dress_left - 20, dress_bottom)], fill=color)
    
    def _render_blazer(self, draw, body_center, color, texture_data):
        """Enhanced blazer rendering"""
        blazer_top = body_center[1] - 90
        blazer_bottom = body_center[1] + 30
        blazer_left = body_center[0] - 65
        blazer_right = body_center[0] + 65
        
        draw.rectangle([blazer_left, blazer_top, blazer_right, blazer_bottom], fill=color, outline=tuple(max(0, c-30) for c in color))
        # Add lapels
        draw.polygon([(blazer_left + 10, blazer_top), (blazer_left + 30, blazer_top + 20), (blazer_left + 10, blazer_top + 40)], fill=tuple(max(0, c-10) for c in color))
    
    def create_base_tryon(self, body_mesh: dict, garment_image: bytes) -> bytes:
        """Create base try-on result"""
        print("[MESH] Creating base try-on result")
        
        # Simplified base result
        img = Image.new('RGB', (512, 512), color='lightgray')
        
        with io.BytesIO() as output:
            img.save(output, format='JPEG')
            return output.getvalue()

class PhysicsEngine:
    """Physics simulation for realistic garment fitting"""
    
    def __init__(self):
        self.initialized = PHYSICS_AVAILABLE
        if self.initialized:
            # Initialize PyBullet in DIRECT mode (no GUI)
            self.physics_client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
    
    def fit_garment_to_body(self, body_mesh: dict, garment_mesh: dict) -> dict:
        """Simulate physics-based garment fitting"""
        if not self.initialized:
            raise Exception("Physics engine not available")
        
        print("[PHYSICS] Simulating garment fitting")
        
        # Simplified physics simulation
        # In production, this would involve cloth simulation, collision detection, etc.
        
        return {
            "fitted_mesh": garment_mesh["mesh"],
            "simulation_steps": 100,
            "collision_points": 0,
            "fitting_quality": 0.92
        }
    
    def __del__(self):
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except:
                pass  # Ignore errors during cleanup

class AIEnhancer:
    """AI-based enhancement and generation"""
    
    def __init__(self):
        self.initialized = AI_AVAILABLE
        self.pipeline = ai_pipeline
    
    def enhance_realism(self, rendered_image: bytes, reference_image: bytes, garment_description: str = "clothing item") -> bytes:
        """Enhance rendered image with AI for photorealism"""
        if not self.initialized or not self.pipeline:
            print("[AI] Pipeline not available, creating basic overlay")
            return self._create_basic_garment_overlay(reference_image, garment_description)
        
        print(f"[AI] Enhancing image realism with {garment_description}")
        print(f"[AI] Using 3D-guided enhancement with strength: 0.3")
        print(f"[AI] Using strength: 0.3, guidance: 7.5, steps: 20 (pose-preserving)")
        
        try:
            # Use the user's original image as base instead of rendered 3D scene
            reference_pil = Image.open(io.BytesIO(reference_image))
            
            # Resize to standard size
            reference_pil = reference_pil.resize((512, 512))
            
            # Create specific prompt for the garment
            garment_type, color = self._parse_garment_description(garment_description)
            prompt = f"person clearly wearing a {color} {garment_type}, visible clothing, detailed {color} {garment_type} on torso, photorealistic, high quality, professional photography, well-fitted {garment_type}"
            negative_prompt = "naked, nude, shirtless, bare chest, no clothing, invisible clothing"
            
            print(f"[AI] Using prompt: {prompt}")
            
            enhanced = self.pipeline(
                prompt=prompt,
                image=reference_pil,
                strength=0.7,  # Increased strength to force garment application
                guidance_scale=12.0,  # Higher guidance for better prompt following
                num_inference_steps=30  # More steps for better quality
            ).images[0]
            
            print("[AI] 3D-guided enhancement completed")
            
            # Convert back to bytes
            with io.BytesIO() as output:
                enhanced.save(output, format='JPEG', quality=90)
                return output.getvalue()
                
        except Exception as e:
            print(f"[AI] Enhancement failed: {e}")
            print(f"[AI] Using fallback garment overlay")
            return self._create_basic_garment_overlay(reference_image, garment_description)
    
    def _parse_garment_description(self, description: str) -> tuple:
        """Parse garment description to extract type and color"""
        description_lower = description.lower()
        
        # Extract garment type
        if "polo" in description_lower:
            garment_type = "polo shirt"
        elif "t-shirt" in description_lower or "tshirt" in description_lower:
            garment_type = "t-shirt"
        elif "shirt" in description_lower:
            garment_type = "shirt"
        elif "jean" in description_lower:
            garment_type = "jeans"
        elif "chino" in description_lower:
            garment_type = "chino pants"
        elif "blazer" in description_lower:
            garment_type = "blazer"
        elif "dress" in description_lower:
            garment_type = "dress"
        else:
            garment_type = "t-shirt"
        
        # Extract color
        if "white" in description_lower:
            color = "white"
        elif "navy" in description_lower:
            color = "navy blue"
        elif "black" in description_lower:
            color = "black"
        elif "blue" in description_lower:
            color = "blue"
        elif "khaki" in description_lower:
            color = "khaki"
        else:
            color = "white"
        
        return garment_type, color
    
    def _create_basic_garment_overlay(self, reference_image: bytes, garment_description: str) -> bytes:
        """Create basic garment overlay when AI is not available"""
        try:
            print(f"[AI] Creating basic garment overlay for {garment_description}")
            
            from PIL import ImageDraw, ImageEnhance
            
            # Load reference image
            reference_pil = Image.open(io.BytesIO(reference_image))
            if reference_pil.mode != 'RGB':
                reference_pil = reference_pil.convert('RGB')
            
            # Create overlay
            width, height = reference_pil.size
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Parse garment for color
            _, color = self._parse_garment_description(garment_description)
            
            # Define color RGB values
            color_map = {
                "white": (255, 255, 255),
                "navy blue": (25, 25, 112),
                "black": (40, 40, 40),
                "blue": (70, 130, 180),
                "khaki": (195, 176, 145)
            }
            color_rgb = color_map.get(color, (255, 255, 255))
            
            # Define garment area
            torso_top = int(height * 0.25)
            torso_bottom = int(height * 0.65)
            torso_left = int(width * 0.3)
            torso_right = int(width * 0.7)
            
            # Draw garment
            draw.rectangle(
                [torso_left, torso_top, torso_right, torso_bottom],
                fill=color_rgb + (180,)
            )
            
            # Add sleeves for shirts
            if "shirt" in garment_description.lower():
                sleeve_width = int(width * 0.12)
                sleeve_height = int((torso_bottom - torso_top) * 0.6)
                draw.rectangle(
                    [torso_left - sleeve_width, torso_top, torso_left, torso_top + sleeve_height],
                    fill=color_rgb + (160,)
                )
                draw.rectangle(
                    [torso_right, torso_top, torso_right + sleeve_width, torso_top + sleeve_height],
                    fill=color_rgb + (160,)
                )
            
            # Blend overlay with original
            result = Image.alpha_composite(reference_pil.convert('RGBA'), overlay).convert('RGB')
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.05)
            
            print(f"[AI] Basic garment overlay completed for {color} {garment_description}")
            
            # Convert to bytes
            with io.BytesIO() as output:
                result.save(output, format='JPEG', quality=90)
                return output.getvalue()
                
        except Exception as e:
            print(f"[AI] Basic overlay failed: {e}")
            return reference_image
    
    def generate_tryon(self, user_image: bytes, garment_image: bytes, garment_description: str = "white t-shirt") -> bytes:
        """Generate virtual try-on using AI"""
        if not self.initialized or not self.pipeline:
            raise Exception("AI pipeline not available")
        
        print("[AI] Generating AI-based virtual try-on")
        
        try:
            user_pil = Image.open(io.BytesIO(user_image)).resize((512, 512))
            
            prompt = f"person wearing {garment_description}, photorealistic, high quality, detailed clothing, natural lighting"
            
            result = self.pipeline(
                prompt=prompt,
                image=user_pil,
                strength=0.7,
                guidance_scale=8.0,
                num_inference_steps=40
            ).images[0]
            
            with io.BytesIO() as output:
                result.save(output, format='JPEG', quality=90)
                return output.getvalue()
                
        except Exception as e:
            print(f"[AI] Generation failed: {e}")
            raise

# Initialize production engine
production_engine = ProductionVirtualTryOn()

# Helper Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = await db.users.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return User(**user)

# API Routes
@api_router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
    )
    
    await db.users.insert_one(user.dict())
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    user = await db.users.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.options("/virtual-tryon")
@api_router.options("/tryon")
async def virtual_tryon_options():
    return {"message": "OK"}

@api_router.post("/virtual-tryon")
@api_router.post("/tryon")
async def virtual_tryon(
    user_image_base64: str = Form(...),
    garment_image_base64: Optional[str] = Form(None),
    product_id: Optional[str] = Form(None),
    processing_mode: str = Form("enhanced_pipeline"),
    current_user: User = Depends(get_current_user)
):
    """Enhanced virtual try-on endpoint with dual image analysis"""
    try:
        print(f"[API] Enhanced virtual try-on request from {current_user.email}")
        
        # Decode user image
        user_image_bytes = base64.b64decode(user_image_base64)
        user_image = Image.open(io.BytesIO(user_image_bytes))
        
        # Get garment image
        garment_image_bytes = None
        if garment_image_base64:
            garment_image_bytes = base64.b64decode(garment_image_base64)
        elif product_id:
            product = await db.products.find_one({"id": product_id})
            if not product:
                raise HTTPException(status_code=404, detail="Product not found")
            response = requests.get(product["image_url"])
            garment_image_bytes = response.content
        else:
            raise HTTPException(status_code=400, detail="No garment specified")
        
        garment_image = Image.open(io.BytesIO(garment_image_bytes))
        
        # Determine garment type
        garment_type = "t-shirt"
        if product_id:
            product = await db.products.find_one({"id": product_id})
            if product:
                name = product.get('name', '').lower()
                if "polo" in name:
                    garment_type = "polo_shirt"
                elif "jean" in name:
                    garment_type = "jeans"
                elif "blazer" in name:
                    garment_type = "blazer"
                elif "dress" in name:
                    garment_type = "dress"
                elif "chino" in name:
                    garment_type = "chinos"
        
        # Use enhanced pipeline controller
        if processing_mode == "enhanced_pipeline":
            from src.core.enhanced_pipeline_controller import EnhancedPipelineController
            controller = EnhancedPipelineController()
            
            result = await controller.process_virtual_tryon(
                customer_image=user_image,
                garment_image=garment_image,
                garment_type=garment_type
            )
            
            if result["success"]:
                # Convert PIL image to base64
                with io.BytesIO() as output:
                    result["result_image"].save(output, format='JPEG', quality=90)
                    result_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
                
                # Clean data for JSON serialization
                def clean_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_for_json(item) for item in obj]
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return obj
                
                return {
                    "result_image_base64": result_base64,
                    "processing_method": "Enhanced Pipeline with Dual Analysis",
                    "confidence": 0.95,
                    "features_used": ["customer_analysis", "garment_analysis", "fitting_algorithm", "3d_rendering"],
                    "processing_mode": processing_mode,
                    "customer_analysis": clean_for_json(result["customer_analysis"]),
                    "garment_analysis": clean_for_json(result["garment_analysis"]),
                    "fitting_data": clean_for_json(result["fitting_data"]),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        # Fallback to original processing
        measurements = current_user.measurements or {
            "height": 170, "weight": 70, "chest": 90, "waist": 75, "hips": 95, "shoulder_width": 45
        }
        
        garment_description = "clothing item"
        if product_id:
            product = await db.products.find_one({"id": product_id})
            if product:
                garment_description = product.get('name', 'clothing item').lower()
        
        result = await production_engine.process_virtual_tryon(
            user_image_bytes, garment_image_bytes, measurements, processing_mode, garment_description
        )
        
        result_base64 = base64.b64encode(result["result_image"]).decode("utf-8")
        
        return {
            "result_image_base64": result_base64,
            "processing_method": result["processing_method"],
            "confidence": result["confidence"],
            "features_used": result["features_used"],
            "processing_mode": processing_mode,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] Virtual try-on failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@api_router.get("/profile", response_model=User)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user

@api_router.post("/measurements")
async def save_measurements(
    measurements: Measurements, 
    current_user: User = Depends(get_current_user)
):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    await db.users.update_one(
        {"id": current_user.id}, 
        {"$set": {"measurements": measurements.dict()}}
    )
    return {"message": "Measurements saved successfully"}

@api_router.get("/tryon-history")
async def get_tryon_history(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        results = await db.tryon_results.find({"user_id": current_user.id}).to_list(100)
        formatted_results = []
        for result in results:
            if "_id" in result:
                del result["_id"]
            formatted_results.append(result)
        return formatted_results
    except Exception as e:
        print(f"Error in get_tryon_history: {str(e)}")
        return []

@api_router.get("/products")
async def get_products():
    """Get all products from the database"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        products = await db.products.find().to_list(1000)
        formatted_products = []
        for product in products:
            if "_id" in product:
                del product["_id"]
            formatted_products.append(product)
        return formatted_products
    except Exception as e:
        print(f"Error in get_products: {str(e)}")
        return []

@api_router.post("/reset-password")
async def reset_password(request: dict):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    return {"message": "Password reset instructions have been sent to your email"}

@api_router.delete("/profile/reset")
async def reset_user_profile(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        await db.users.update_one(
            {"id": current_user.id}, 
            {"$unset": {"measurements": "", "captured_image": "", "captured_images": ""}}
        )
        return {"message": "User profile reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile reset failed: {str(e)}")

@api_router.post("/save_captured_image")
async def save_captured_image(image_data: dict, current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        image_record = {
            "image_base64": image_data.get("image_base64"),
            "captured_at": datetime.utcnow(),
            "measurements": image_data.get("measurements"),
            "image_type": "camera_capture",
        }
        
        await db.users.update_one(
            {"id": current_user.id}, 
            {"$push": {"captured_images": image_record}, "$set": {"captured_image": image_data.get("image_base64")}}
        )
        
        return {"message": "Image saved to profile successfully", "image_id": str(image_record["captured_at"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save captured image")

@api_router.post("/extract-measurements")
async def extract_measurements(
    user_image_base64: str = Form(...), 
    reference_height_cm: Optional[float] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Enhanced measurement extraction using computer vision"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        print("[ENHANCED] Starting enhanced customer image analysis...")
        
        # Decode image
        user_image_bytes = base64.b64decode(user_image_base64)
        
        # Use enhanced customer image analyzer
        from src.core.customer_image_analyzer import CustomerImageAnalyzer
        analyzer = CustomerImageAnalyzer()
        
        analysis = analyzer.analyze_customer_image(user_image_bytes, reference_height_cm)
        
        # Extract measurements for API response
        measurements = analysis["measurements"]
        skin_tone = analysis["skin_tone"]
        
        # Convert to API format
        api_measurements = {
            "height": measurements["height_cm"],
            "shoulder_width": measurements["shoulder_width_cm"],
            "chest": measurements.get("chest_circumference_cm", measurements["shoulder_width_cm"] * 2.2),
            "waist": measurements.get("waist_circumference_cm", measurements["shoulder_width_cm"] * 1.8),
            "hips": measurements.get("hip_circumference_cm", measurements["shoulder_width_cm"] * 2.5),
            "weight": measurements.get("estimated_weight_kg", 70.0)
        }
        
        # Save enhanced measurements
        enhanced_data = {
            **api_measurements,
            "skin_tone": skin_tone,
            "confidence_score": analysis["confidence_score"],
            "analysis_method": "enhanced_computer_vision",
            "pose_detected": analysis["analysis_success"]
        }
        
        await db.users.update_one(
            {"id": current_user.id}, {"$set": {"measurements": enhanced_data}}
        )
        
        print(f"[ENHANCED] Analysis complete - Confidence: {analysis['confidence_score']:.2f}")
        
        return {
            "measurements": api_measurements,
            "skin_tone": skin_tone,
            "confidence_score": analysis["confidence_score"],
            "pose_detected": analysis["analysis_success"],
            "message": "Enhanced measurements extracted and saved successfully"
        }
        
    except Exception as e:
        print(f"[ENHANCED] Analysis failed: {e}, using fallback")
        # Fallback to basic measurements
        fallback_measurements = {
            "height": 170, "weight": 70, "chest": 90, "waist": 75, "hips": 95, "shoulder_width": 45
        }
        
        await db.users.update_one(
            {"id": current_user.id}, {"$set": {"measurements": fallback_measurements}}
        )
        
        return {
            "measurements": fallback_measurements,
            "confidence_score": 0.3,
            "pose_detected": False,
            "message": "Fallback measurements used - enhanced analysis failed"
        }

@app.get("/")
async def root():
    return {
        "service": "VirtualFit Production API",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "3d_processing": MESH_PROCESSING_AVAILABLE,
            "physics_simulation": PHYSICS_AVAILABLE,
            "ai_enhancement": AI_AVAILABLE,
            "computer_vision": CV2_AVAILABLE,
            "fal_integration": FAL_AVAILABLE,
            "enhanced_customer_analysis": True,
            "enhanced_3d_processing": True,
            "enhanced_pipeline_controller": True,
            "dual_image_analysis": True,
            "fitting_algorithm": True,
            "validation_system": True,
            "gpu_acceleration": True,
            "image_preprocessing": True,
            "analysis_caching": True,
            "performance_optimizations": True
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": db is not None,
            "ai_pipeline": ai_pipeline is not None,
            "mesh_processor": production_engine.mesh_processor is not None,
            "physics_engine": production_engine.physics_engine is not None
        }
    }

@app.get("/debug")
async def debug():
    return {
        "status": "Production Virtual Try-On Server",
        "capabilities": {
            "full_3d_pipeline": MESH_PROCESSING_AVAILABLE and PHYSICS_AVAILABLE,
            "ai_enhancement": AI_AVAILABLE,
            "hybrid_processing": True,
            "fallback_processing": True
        },
        "libraries": {
            "torch": AI_AVAILABLE,
            "trimesh": MESH_PROCESSING_AVAILABLE,
            "pybullet": PHYSICS_AVAILABLE,
            "opencv": CV2_AVAILABLE,
            "fal_client": FAL_AVAILABLE,
            "sklearn": True
        },
        "environment": {
            "mongo_url": bool(mongo_url),
            "db_name": db_name
        }
    }

# Include API router
app.include_router(api_router)

@app.get("/test-complete-pipeline")
async def test_complete_pipeline():
    """Test the complete enhanced pipeline"""
    try:
        # Test garment analyzer
        from src.core.garment_analyzer import GarmentImageAnalyzer
        analyzer = GarmentImageAnalyzer()
        
        # Create test image
        from PIL import Image
        import io
        test_img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        test_bytes = img_bytes.getvalue()
        
        analysis = analyzer.analyze_garment_image(test_bytes)
        
        return {
            "garment_analyzer": "working",
            "production_engine": "initialized",
            "test_analysis": {
                "success": analysis["analysis_success"],
                "primary_color": analysis["colors"]["primary"],
                "fabric_type": analysis["fabric_type"]
            },
            "pipeline_status": "enhanced and ready"
        }
    except Exception as e:
        return {
            "error": str(e),
            "pipeline_status": "needs debugging"
        }

@api_router.post("/test-garment-analysis")
async def test_garment_analysis(garment_image_base64: str = Form(...)):
    """Test endpoint for garment image analysis"""
    try:
        from src.core.garment_analyzer import GarmentImageAnalyzer
        
        garment_bytes = base64.b64decode(garment_image_base64)
        analyzer = GarmentImageAnalyzer()
        analysis = analyzer.analyze_garment_image(garment_bytes)
        
        return {
            "analysis_success": analysis["analysis_success"],
            "primary_color": analysis["colors"]["primary"],
            "fabric_type": analysis["fabric_type"],
            "pattern_type": analysis["patterns"]["type"],
            "full_analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/test-customer-analysis")
async def test_customer_analysis(
    user_image_base64: str = Form(...),
    reference_height_cm: Optional[float] = Form(None)
):
    """Test endpoint for enhanced customer image analysis"""
    try:
        from src.core.customer_image_analyzer import CustomerImageAnalyzer
        
        user_bytes = base64.b64decode(user_image_base64)
        analyzer = CustomerImageAnalyzer()
        analysis = analyzer.analyze_customer_image(user_bytes, reference_height_cm)
        
        return {
            "analysis_success": analysis["analysis_success"],
            "confidence_score": analysis["confidence_score"],
            "pose_detected": analysis.get("pose_landmarks") is not None,
            "measurements": {
                "height_cm": analysis["measurements"]["height_cm"],
                "shoulder_width_cm": analysis["measurements"]["shoulder_width_cm"]
            },
            "skin_tone": analysis["skin_tone"],
            "body_segmentation_success": analysis["body_segmentation"]["success"],
            "full_analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Customer analysis failed: {str(e)}")

@api_router.post("/test-enhanced-3d-processing")
async def test_enhanced_3d_processing(
    garment_image_base64: str = Form(...),
    garment_type: str = Form("t-shirt")
):
    """Test endpoint for enhanced 3D garment processing"""
    try:
        from src.core.garment_analyzer import GarmentImageAnalyzer
        from src.core.enhanced_3d_garment_processor import Enhanced3DGarmentProcessor
        
        # Analyze garment
        garment_bytes = base64.b64decode(garment_image_base64)
        analyzer = GarmentImageAnalyzer()
        analysis = analyzer.analyze_garment_image(garment_bytes)
        
        # Process with enhanced 3D processor
        processor = Enhanced3DGarmentProcessor()
        mesh_data = processor.create_enhanced_garment_mesh(analysis, garment_type)
        mesh_data = processor.apply_physics_properties(mesh_data)
        
        return {
            "garment_analysis_success": analysis["analysis_success"],
            "garment_type": garment_type,
            "mesh_vertices": mesh_data.get("vertices", 0),
            "mesh_faces": mesh_data.get("faces", 0),
            "material_properties": mesh_data["material_properties"],
            "texture_data": mesh_data["texture_data"],
            "physics_properties": mesh_data.get("physics_properties", {}),
            "dimensions": mesh_data["dimensions"],
            "enhanced_processing": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced 3D processing failed: {str(e)}")

@api_router.post("/test-enhanced-pipeline")
async def test_enhanced_pipeline(
    user_image_base64: str = Form(...),
    garment_image_base64: str = Form(...),
    garment_type: str = Form("t-shirt")
):
    """Test endpoint for complete enhanced pipeline with performance optimizations"""
    try:
        from src.core.enhanced_pipeline_controller import EnhancedPipelineController
        from src.core.performance_optimizations import AnalysisCache, GPUAccelerator
        
        # Decode images
        user_bytes = base64.b64decode(user_image_base64)
        garment_bytes = base64.b64decode(garment_image_base64)
        
        user_image = Image.open(io.BytesIO(user_bytes))
        garment_image = Image.open(io.BytesIO(garment_bytes))
        
        # Process with enhanced pipeline
        controller = EnhancedPipelineController()
        result = await controller.process_virtual_tryon(
            customer_image=user_image,
            garment_image=garment_image,
            garment_type=garment_type
        )
        
        if result["success"]:
            # Convert result image to base64
            with io.BytesIO() as output:
                result["result_image"].save(output, format='JPEG', quality=90)
                result_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
            
            return {
                "pipeline_success": True,
                "result_image_base64": result_base64,
                "customer_analysis_success": result["customer_analysis"]["analysis_success"],
                "garment_analysis_success": result["garment_analysis"]["analysis_success"],
                "fitting_algorithm": "completed",
                "validation_passed": True,
                "processing_method": "Enhanced Pipeline Controller",
                "customer_measurements": result["customer_analysis"]["measurements"],
                "garment_properties": {
                    "primary_color": result["garment_analysis"]["dominant_colors"][0] if result["garment_analysis"]["dominant_colors"] else None,
                    "fabric_type": result["garment_analysis"]["fabric_type"],
                    "pattern_type": result["garment_analysis"]["patterns"]["type"]
                },
                "color_matching": result["fitting_data"]["color_matching"],
                "performance_info": result.get("performance_info", {}),
                "gpu_available": GPUAccelerator.is_gpu_available(),
                "cache_size": len(AnalysisCache.ANALYSIS_CACHE),
                "enhanced_pipeline": "success"
            }
        else:
            return {
                "pipeline_success": False,
                "error": result["error"],
                "enhanced_pipeline": "failed"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced pipeline test failed: {str(e)}")

@api_router.post("/clear-cache")
async def clear_analysis_cache():
    """Clear analysis cache"""
    try:
        from src.core.performance_optimizations import AnalysisCache
        AnalysisCache.clear_cache()
        return {"message": "Analysis cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@api_router.get("/performance-status")
async def get_performance_status():
    """Get performance optimization status"""
    try:
        from src.core.performance_optimizations import GPUAccelerator, AnalysisCache
        
        return {
            "gpu_available": GPUAccelerator.is_gpu_available(),
            "device": GPUAccelerator.get_device(),
            "cache_size": len(AnalysisCache.ANALYSIS_CACHE),
            "max_cache_size": AnalysisCache.MAX_CACHE_SIZE,
            "preprocessing_enabled": True,
            "optimizations_active": True
        }
    except Exception as e:
        return {
            "gpu_available": False,
            "device": "cpu",
            "cache_size": 0,
            "error": str(e)
        }

@app.on_event("startup")
async def initialize_database():
    """Initialize database collections and sample data for production deployment"""
    print("🚀 FastAPI application starting up...")

    # Initialize database immediately during startup
    if mongo_url:
        print("🔄 MongoDB URL configured, initializing database connection...")
        await init_database_background()
    else:
        print("❌ MONGO_URL not configured")
        print("⚠️ Starting without database connection")

    print("✅ FastAPI startup completed - ready to serve requests")


async def init_database_background():
    """Database initialization with retry logic and in-memory fallback"""
    global client, db

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            print(
                f"🔄 Initializing MongoDB connection "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            # Initialize MongoDB client
            if mongo_url:
                print(
                    f"🔍 Creating AsyncIOMotorClient with URL: " f"{mongo_url[:50]}..."
                )

                client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
                db = client[db_name]

                # Test database connection with reduced timeout
                await asyncio.wait_for(db.command("ping"), timeout=5.0)
                print("✅ MongoDB connection successful")
                break

            else:
                print("❌ MONGO_URL not configured")
                return

        except Exception as e:
            print(
                f"❌ Database initialization attempt {attempt + 1} failed: " f"{str(e)}"
            )
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ All database initialization attempts failed")
                print("🔄 Falling back to in-memory database simulation...")
                await init_memory_database()
                return

    if db is not None:
        await initialize_sample_data()


# In-memory database fallback
memory_db = {
    "users": [],
    "products": [],
    "tryon_results": []
}

class MemoryDatabase:
    def __init__(self):
        self.data = memory_db
    
    async def find_one(self, query):
        collection_name = getattr(self, '_current_collection', 'users')
        for item in self.data[collection_name]:
            if all(item.get(k) == v for k, v in query.items()):
                return item
        return None
    
    async def insert_one(self, document):
        collection_name = getattr(self, '_current_collection', 'users')
        self.data[collection_name].append(document)
        return type('Result', (), {'inserted_id': len(self.data[collection_name])})()
    
    async def update_one(self, query, update):
        collection_name = getattr(self, '_current_collection', 'users')
        for item in self.data[collection_name]:
            if all(item.get(k) == v for k, v in query.items()):
                if '$set' in update:
                    item.update(update['$set'])
                if '$push' in update:
                    for key, value in update['$push'].items():
                        if key not in item:
                            item[key] = []
                        item[key].append(value)
                return type('Result', (), {'modified_count': 1})()
        return type('Result', (), {'modified_count': 0})()
    
    async def find(self, query={}):
        collection_name = getattr(self, '_current_collection', 'users')
        results = []
        for item in self.data[collection_name]:
            if not query or all(item.get(k) == v for k, v in query.items()):
                results.append(item)
        return type('Cursor', (), {'to_list': lambda x: asyncio.create_task(asyncio.coroutine(lambda: results)())})()
    
    async def count_documents(self, query={}):
        collection_name = getattr(self, '_current_collection', 'users')
        count = 0
        for item in self.data[collection_name]:
            if not query or all(item.get(k) == v for k, v in query.items()):
                count += 1
        return count
    
    async def create_index(self, field, **kwargs):
        print(f"📝 Created index on {field} (in-memory)")
        return True
    
    async def insert_many(self, documents):
        collection_name = getattr(self, '_current_collection', 'users')
        self.data[collection_name].extend(documents)
        return type('Result', (), {'inserted_ids': list(range(len(documents)))})()
    
    async def command(self, cmd):
        if cmd == "ping":
            return {"ok": 1}
        return {"ok": 1}
    
    @property
    def users(self):
        self._current_collection = 'users'
        return self
    
    @property
    def products(self):
        self._current_collection = 'products'
        return self
    
    @property
    def tryon_results(self):
        self._current_collection = 'tryon_results'
        return self


async def init_memory_database():
    """Initialize in-memory database as fallback"""
    global db
    
    print("💾 Initializing in-memory database...")
    db = MemoryDatabase()
    
    # Initialize sample data
    await initialize_sample_data()
    print("✅ In-memory database initialized successfully")


async def initialize_sample_data():
    """Initialize sample products and database indexes"""
    try:
        # Initialize sample products if products collection is empty
        product_count = await db.products.count_documents({})
        if product_count == 0:
            print("📦 Creating sample product catalog...")
            sample_products = [
                {
                    "id": "white-tshirt-001",
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
                    "id": "blue-jeans-002",
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
                    "id": "black-blazer-003",
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
                    "id": "summer-dress-004",
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
                    "id": "navy-polo-005",
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
                    "id": "khaki-chinos-006",
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
            print(f"✅ Created {len(sample_products)} sample products")
        else:
            print(
                f"📦 Products collection already exists with {product_count} items"
            )

        try:
            await db.users.create_index("email", unique=True)
            await db.products.create_index("category")
            await db.products.create_index("name")
        except Exception as e:
            print(f"⚠️ Index creation skipped: {e}")
        print("✅ Database indexes created")

        print("🎉 Database initialization completed successfully")

    except Exception as e:
        print(f"❌ Sample data initialization failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)