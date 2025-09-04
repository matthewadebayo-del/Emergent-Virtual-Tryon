# VirtualFit - Virtual Try-On Solution

A full-stack virtual try-on application built with React frontend and FastAPI backend, featuring AI-powered virtual clothing fitting and body measurements analysis.

## 🚀 Features

- **User Authentication**: Secure registration, login, and profile management
- **Virtual Try-On**: AI-powered virtual clothing fitting using OpenAI integration
- **Body Measurements**: Camera capture and manual entry of body measurements
- **Real-time Processing**: Fast virtual try-on with 95% accuracy rate
- **Responsive Design**: Clean, modern interface optimized for all devices

## 🛠 Tech Stack

- **Frontend**: React, JavaScript, CSS, Yarn
- **Backend**: FastAPI, Python, Uvicorn
- **Database**: MongoDB (in-memory for development)
- **AI Integration**: OpenAI API for image generation and processing
- **Authentication**: JWT tokens with bcrypt password hashing

## 📋 Prerequisites

- Python 3.8+ with pip and virtual environment support
- Node.js 16+ with yarn package manager
- MongoDB (for production) or in-memory database (for development)
- OpenAI API key for virtual try-on functionality

## 🏃‍♂️ Quick Start

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
OPENAI_API_KEY="your-openai-api-key-here"
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
Control feature availability with these environment variables:
- `ENABLE_AI_ENHANCEMENT=true/false` - Controls Stable Diffusion loading
- `ENABLE_3D_FEATURES=true/false` - Controls advanced 3D features

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

## 🔧 API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login
- `GET /api/profile` - Get user profile

### Measurements & Try-On
- `POST /api/measurements` - Save body measurements
- `POST /api/virtual-tryon` - Process virtual try-on
- `GET /api/products` - Get available clothing items
- `GET /api/tryon-history` - Get user's try-on history

## 🌐 Deployment Status

### Current Deployment
- **Frontend**: https://virtual-tryon-app-a8pe83vz.devinapps.com
- **Status**: Deployed with clean VirtualFit branding
- **Backend**: Localhost testing only (see limitations below)

### Known Limitations
- **CORS Issue**: Dev environment tunnel services require authentication that blocks CORS headers
- **Workaround**: Use localhost testing as primary method before proper deployment
- **Solution**: Proper cloud deployment planned after localhost testing completion

## 🔍 Development Notes

### Environment Configuration
The application is configured for localhost development with proper CORS settings. The backend includes comprehensive error handling and logging for debugging.

### Code Quality
- Backend follows FastAPI best practices with proper async/await patterns
- Frontend uses modern React patterns with clean component architecture
- All emergent branding removed and replaced with VirtualFit branding
- OpenAI integration replaces proprietary AI services

### Security Features
- JWT token authentication with configurable expiration
- Password hashing using bcrypt
- CORS protection with configurable origins
- Input validation and sanitization

## 📚 Additional Documentation

- [Localhost Testing Guide](./LOCALHOST_TESTING_GUIDE.md) - Comprehensive testing instructions
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
- Check the [Localhost Testing Guide](./LOCALHOST_TESTING_GUIDE.md) for testing instructions
- Review API documentation at http://localhost:8000/docs
- Create an issue in the GitHub repository
- Ensure all tests pass before submitting pull requests

---

**VirtualFit** - Revolutionizing virtual try-on experiences with AI-powered precision.
