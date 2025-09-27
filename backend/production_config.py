"""
Production configuration for VirtualFit
"""
import os
from pathlib import Path

# Production settings
PRODUCTION_MODE = True
DEBUG = False

# Database configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "virtualfit_production")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "virtualfit-production-secret-key")
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# CORS settings for production
CORS_ORIGINS = [
    "http://localhost:3000",
    "https://virtual-tryon-app-a8pe83vz.devinapps.com",
    "*"  # Remove in strict production
]

# Feature flags
ENABLE_AI_ENHANCEMENT = os.getenv("ENABLE_AI_ENHANCEMENT", "true").lower() == "true"
ENABLE_3D_FEATURES = os.getenv("ENABLE_3D_FEATURES", "true").lower() == "true"
USE_COMPREHENSIVE_TRYON = os.getenv("USE_COMPREHENSIVE_TRYON", "true").lower() == "true"

# Performance settings
MAX_WORKERS = 4
WORKER_TIMEOUT = 300
MAX_REQUEST_SIZE = 50 * 1024 * 1024  # 50MB

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/virtualfit.log"

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Redis configuration (for future scaling)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# API rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# File upload limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]

# Cache settings
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1000

print(f"[CONFIG] Production mode: {PRODUCTION_MODE}")
print(f"[CONFIG] AI Enhancement: {ENABLE_AI_ENHANCEMENT}")
print(f"[CONFIG] 3D Features: {ENABLE_3D_FEATURES}")
print(f"[CONFIG] Comprehensive Try-on: {USE_COMPREHENSIVE_TRYON}")
print(f"[CONFIG] Database: {DB_NAME}")