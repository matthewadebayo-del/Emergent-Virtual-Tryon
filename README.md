# VirtualFit - Virtual Try-On Solution

A full-stack virtual try-on application built with React frontend and FastAPI backend, featuring AI-powered virtual clothing fitting and body measurements analysis.

## 🚀 Features

- **User Authentication**: Secure registration, login, and profile management
- **Comprehensive Virtual Try-On**: Region-based virtual clothing fitting with 5 garment type support (TOP, BOTTOM, SHOES, DRESS, OUTERWEAR)
- **Garment Image Analysis**: Automatic extraction of colors, textures, patterns, and fabric properties
- **Customer Image Analysis**: Pose detection, body measurements, and skin tone analysis
- **3D Garment Processing**: Realistic mesh creation with material-specific properties
- **Body Measurements**: Camera capture and manual entry of 29+ measurement items
- **Real-time Processing**: GPU-accelerated pipeline with caching for optimal performance
- **Responsive Design**: Clean, modern interface optimized for all devices

## 🛠 Tech Stack

- **Frontend**: React, JavaScript, CSS, Yarn
- **Backend**: FastAPI, Python, Uvicorn
- **Database**: MongoDB (in-memory for development)
- **Computer Vision**: OpenCV, MediaPipe, scikit-learn
- **3D Processing**: Advanced mesh processing with physics simulation
- **AI Integration**: FASHN API for virtual try-on with Vertex AI fallback
- **Performance**: GPU acceleration, parallel processing, intelligent caching
- **Authentication**: JWT tokens with bcrypt password hashing

## 📋 Prerequisites

- Python 3.8+ with pip and virtual environment support
- Node.js 16+ with yarn package manager
- MongoDB (for production) or in-memory database (for development)
- FASHN API key for virtual try-on functionality

## 🏃♂️ Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/matthewadebayo-del/Emergent-Virtual-Tryon.git
cd Emergent-Virtual-Tryon
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file in backend directory:
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="http://localhost:3000,*"
FASHN_API_KEY="your-fashn-api-key-here"
```

Start backend server:
```bash
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
yarn install
```

Create `.env` file in frontend directory:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
WDS_SOCKET_PORT=443
```

Start frontend development server:
```bash
yarn start
```

### 4. Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🐳 Docker Deployment Options

### Minimal Deployment (~3GB)
- Basic virtual try-on functionality
- No AI enhancement or physics simulation
- Suitable for development and basic testing
```bash
docker build -f backend/Dockerfile.minimal -t virtualfit-backend-minimal .
```

### Production Deployment (~5GB)  
- Complete 3D virtual try-on with CPU-optimized PyTorch
- MediaPipe body reconstruction, physics simulation, and AI enhancement
- Recommended for production deployment (Cloud Run compatible)
```bash
docker build -f backend/Dockerfile.production -t virtualfit-backend-production .
```

### Optimized Deployment (~8GB)  
- Core 3D features with conditional AI loading
- MediaPipe body reconstruction and basic rendering
- Alternative for moderate resource environments
```bash
docker build -f backend/Dockerfile.optimized -t virtualfit-backend-optimized .
```

### Full Deployment (~20GB)
- All features including GPU-optimized PyTorch and development tools
- Complete 3D virtual try-on with maximum performance
- Requires high-memory environments
```bash
docker build -f backend/Dockerfile -t virtualfit-backend-full .
```

### Environment Variables
Control API integration with these environment variables:
- `FASHN_API_KEY` - Primary virtual try-on service
- `VERTEX_AI_ENABLED=true/false` - Controls Vertex AI fallback
- `SUPPORT_CONTACT_EMAIL` - Contact for service issues

## 📖 Testing Guide

For comprehensive testing instructions, see [LOCALHOST_TESTING_GUIDE.md](./LOCALHOST_TESTING_GUIDE.md).

### Quick Test Flow

1. **Register Demo User**:
   - Navigate to http://localhost:3000/register
   - Use: `demo@virtualfit.com` / `demo123`

2. **Test Authentication**:
   - Login with demo credentials
   - Verify dashboard access and logout functionality

3. **Test Core Features**:
   - Camera capture for measurements
   - Upload & try-on functionality
   - Manual measurements entry
   - Comprehensive virtual try-on with different garment types
   - Region-based processing validation

4. **Test Comprehensive Try-On**:
   - Run `python backend/test_comprehensive_tryon.py` for garment type testing
   - Test configurations include: T-shirts, Jeans, Sneakers, Dresses, Full Outfits
   - Validate region preservation (face, arms, background) and modification (clothing areas)

## 🔧 API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login
- `GET /api/profile` - Get user profile

### Measurements & Try-On
- `POST /api/measurements` - Save body measurements (29+ items)
- `POST /api/virtual-tryon` - Virtual try-on using FASHN API with fallback system
- `GET /api/products` - Get available clothing items
- `GET /api/tryon-history` - Get user's try-on history

### Virtual Try-On System
- **Primary**: FASHN API integration with job polling system
- **Fallback**: Vertex AI processing when FASHN is unavailable
- **Error Handling**: Returns identical before/after images with support message when all systems fail
- **Reliability**: Multi-tier fallback ensures consistent user experience
- **Status Messages**: Clear communication when virtual try-on services are temporarily unavailable

## 🌐 Deployment Status

### Current Deployment
- **Frontend**: https://virtual-tryon-app-a8pe83vz.devinapps.com
- **Status**: Deployed with clean VirtualFit branding
- **Backend**: Production server requires update (see [PRODUCTION_UPDATE_INSTRUCTIONS.md](./PRODUCTION_UPDATE_INSTRUCTIONS.md))

### Known Issues
- **Production Server**: Running outdated SAFE mode instead of comprehensive try-on system
- **Solution**: Update production server code and restart (detailed instructions provided)

## 🎆 Recent Major Enhancements

### FASHN API Integration (Latest - 2024-12-27)
- **Primary Service**: Integrated FASHN API (api.fashn.ai) as main virtual try-on provider
- **Job Polling System**: Implemented asynchronous job submission and status polling
- **Multi-Tier Fallback**: FASHN → Vertex AI → Error handling with identical images
- **Reliable Experience**: Ensures users always receive response even when services are down
- **Clear Communication**: Status messages inform users when virtual try-on is temporarily unavailable
- **Status**: Production-ready with comprehensive fallback system

### L.L.Bean Product Catalog Integration (Previous - 2024-12-27)
- **Comprehensive Catalog**: 12 L.L.Bean products with multiple color variations
- **Product Categories**: Men's and women's shirts (short/long sleeve), pants, and accessories
- **Color Options**: White, navy, red, pink, blue, khaki variations for each product
- **Persistent Storage**: Products automatically loaded into database on server startup
- **Status**: Production-ready product catalog system

### Persistent Measurement Storage (Previous - 2024-12-27)
- **27+ Body Measurements**: Comprehensive measurement system with permanent storage
- **Image Hash Tracking**: Smart re-extraction only when new images are detected
- **Auto-Update System**: Existing databases automatically updated with measurement fields
- **Performance Optimization**: Avoids unnecessary measurement re-extraction
- **Status**: Production-ready persistent measurement system

### Production Server Update Required (Previous)
- **Issue**: Production server using outdated SAFE mode causing identical before/after images
- **Root Cause**: Server running old code with `USE_COMPREHENSIVE_TRYON = False`
- **Solution**: Update production server code and restart to enable comprehensive try-on system
- **Status**: Instructions provided in [PRODUCTION_UPDATE_INSTRUCTIONS.md](./PRODUCTION_UPDATE_INSTRUCTIONS.md)

### Comprehensive Virtual Try-On System (Previous Update)
- **Complete SAFE Mode Replacement**: Implemented 12-step comprehensive region-based virtual try-on system
- **Region-Based Processing**: Precise garment type detection with 5 supported categories (TOP, BOTTOM, SHOES, DRESS, OUTERWEAR)
- **Advanced Landmark Validation**: Confidence-based pose detection with 0.7+ threshold requirements
- **Enhanced Error Handling**: Comprehensive try-catch blocks with graceful fallback to original images
- **Performance Monitoring**: Real-time processing time tracking with performance threshold alerts
- **Intelligent Configuration**: Dynamic garment type mapping with combination support (outfits, sets)

### System Overhaul (December 2024)
- **Fixed Critical Rendering Issue**: Resolved garment visibility problem in final virtual try-on images
- **Computer Vision Integration**: Implemented comprehensive image analysis for both garments and customers
- **Advanced 3D Processing**: Built realistic garment mesh creation with material-specific properties
- **Performance Optimization**: Added GPU acceleration with ~40% faster processing and intelligent caching
- **Enhanced Pipeline**: Created parallel processing controller with validation and fitting algorithms

### Core System Components
- **CompleteGarmentReplacement**: Revolutionary 6-step garment replacement system with true clothing removal
- **ComprehensiveRegionTryOn**: Advanced region-based virtual try-on processor with 5 garment type support
- **ProcessingResult**: Structured result handling with success/failure states and detailed metadata
- **GarmentType**: Enum-based garment classification system (TOP, BOTTOM, SHOES, DRESS, OUTERWEAR)
- **GarmentImageAnalyzer**: Extracts colors, textures, patterns, and fabric properties from garment images
- **CustomerImageAnalyzer**: Performs pose detection, measurements extraction, and skin tone analysis
- **Enhanced3DGarmentProcessor**: Creates realistic 3D meshes with material-specific properties
- **EnhancedPipelineController**: Main orchestrator with complete replacement integration and validation
- **PerformanceOptimizations**: GPU acceleration and caching utilities

### Technical Architecture
- **FASHN API Integration**: Primary virtual try-on service with job polling system
  1. **Job Submission**: Asynchronous job creation with customer and garment images
  2. **Status Polling**: Real-time job status monitoring with timeout handling
  3. **Result Processing**: Automatic result retrieval and image processing
  4. **Error Handling**: Comprehensive exception handling with fallback triggers
- **Multi-Tier Fallback System**: Ensures reliable user experience
  1. **Primary**: FASHN API processing with job polling
  2. **Secondary**: Vertex AI processing when FASHN unavailable
  3. **Tertiary**: Identical before/after images with support message
- **Persistent Measurement Storage**: 27+ body measurements with image hash tracking
- **Smart Re-extraction**: Measurements updated only when new images detected
- **L.L.Bean Product Catalog**: Comprehensive product database with color variations
- **Authentication System**: JWT token management with protected endpoints
- **Performance Optimization**: Caching and smart processing to minimize API calls

## 🔍 Development Notes

### Environment Configuration
The application is configured for localhost development with proper CORS settings. The backend includes comprehensive error handling and logging for debugging.

### Code Quality
- Backend follows FastAPI best practices with proper async/await patterns
- Frontend uses modern React patterns with clean component architecture
- All emergent branding removed and replaced with VirtualFit branding
- Complete garment replacement system with professional-grade image processing
- Computer vision integration using OpenCV, MediaPipe, and scikit-learn
- True garment replacement instead of filter-like blending

### Security Features
- JWT token authentication with configurable expiration
- Password hashing using bcrypt
- CORS protection with configurable origins
- Input validation and sanitization

### Performance Improvements
- **FASHN API Integration**: Professional-grade virtual try-on processing
- **Asynchronous Processing**: Job polling system prevents blocking operations
- **Smart Measurement Caching**: Image hash tracking avoids unnecessary re-extraction
- **Multi-Tier Fallback**: Ensures consistent response times even during service outages
- **Persistent Storage**: Measurements and products cached for optimal performance

## 📚 Additional Documentation

- [Localhost Testing Guide](./LOCALHOST_TESTING_GUIDE.md) - Comprehensive testing instructions
- [Production Update Instructions](./PRODUCTION_UPDATE_INSTRUCTIONS.md) - Fix for SAFE mode issue
- [AI Providers Guide](./AI_PROVIDERS_GUIDE.md) - AI integration documentation
- [AWS Deployment Guide](./AWS_DEPLOYMENT_GUIDE.md) - Cloud deployment instructions
- [GCP Firebase Guide](./GCP_FIREBASE_DEPLOYMENT_GUIDE.md) - Firebase deployment options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and test thoroughly
4. Run linting: `black backend/` and `isort backend/`
5. Commit your changes: `git commit -m "Add your feature"`
6. Push to the branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:
- Check the [Production Update Instructions](./PRODUCTION_UPDATE_INSTRUCTIONS.md) for server update issues
- Check the [Localhost Testing Guide](./LOCALHOST_TESTING_GUIDE.md) for testing instructions
- Review API documentation at http://localhost:8000/docs
- Create an issue in the GitHub repository
- Ensure all tests pass before submitting pull requests

---

**VirtualFit** - Revolutionizing virtual try-on experiences with **FASHN API integration and reliable fallback systems**.

**Latest Achievement**: FASHN API integration with multi-tier fallback system ensuring consistent user experience even during service outages!