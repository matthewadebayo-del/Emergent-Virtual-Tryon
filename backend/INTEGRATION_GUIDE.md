# VirtualFit Integration Guide

## Overview
VirtualFit now supports dual-mode operation:
1. **Standalone Web Application** - Original React frontend
2. **E-commerce API Integration** - RESTful API for external sites

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start all services
python start_services.py
```

## API Integration

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
Include API key in request:
```json
{
  "api_key": "your-api-key-here"
}
```

### Endpoints

#### 1. Async Processing (Recommended)
```http
POST /api/v1/tryon/process
Content-Type: application/json

{
  "api_key": "your-api-key",
  "customer_image_url": "https://example.com/customer.jpg",
  "garment_image_url": "https://example.com/garment.jpg",
  "product_info": {
    "name": "Classic White T-Shirt",
    "category": "TOP",
    "color": "white",
    "size": "M"
  },
  "webhook_url": "https://your-site.com/webhook"
}
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "task_id": "celery-task-id"
}
```

#### 2. Check Status
```http
GET /api/v1/tryon/status/{job_id}
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "progress": 100,
  "result_url": "base64-image-data"
}
```

#### 3. Get Result
```http
GET /api/v1/tryon/result/{job_id}
```

#### 4. Sync Processing (Real-time)
```http
POST /api/v1/tryon/sync
```
Same request format as async, returns immediate result.

### Webhook Notifications

When processing completes, webhook receives:
```json
{
  "task_id": "celery-task-id",
  "status": "completed",
  "result": {
    "success": true,
    "result_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "processing_time": 2.3,
    "service_used": "vertex_ai"
  }
}
```

## Service Architecture

### Components
- **FastAPI Server** (Port 8000) - Main API
- **Celery Worker** - Background processing
- **Redis** (Port 6379) - Task queue & caching
- **Flower** (Port 5555) - Task monitoring
- **MongoDB** (Port 27017) - Data storage

### Processing Pipeline
1. **Image Download** - Fetch customer & garment images
2. **Customer Analysis** - Pose detection, measurements
3. **Garment Analysis** - Color, texture, fabric properties
4. **Virtual Try-On** - AI-powered garment replacement
5. **Result Delivery** - Base64 image + metadata

## Configuration

### Environment Variables
```env
# Redis
REDIS_URL=redis://localhost:6379/0

# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=virtualfit_production

# AI Services
OPENAI_API_KEY=your-openai-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
VERTEX_AI_PROJECT_ID=your-project-id
FASHN_API_KEY=your-fashn-key

# Features
ENABLE_AI_ENHANCEMENT=true
ENABLE_3D_FEATURES=true
```

### API Keys Setup
1. **Vertex AI**: Create service account in Google Cloud Console
2. **FASHN**: Register at fashn.ai for API key
3. **OpenAI**: Get API key from OpenAI platform

## Error Handling

### Common Errors
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Job ID not found
- `400 Bad Request` - Invalid image URLs or format
- `500 Internal Server Error` - Processing failure

### Retry Logic
- Automatic failover between AI services
- Webhook retry (3 attempts)
- Task timeout: 300 seconds

## Performance

### Optimization Features
- **Redis Caching** - Results cached for 1 hour
- **Parallel Processing** - Multiple Celery workers
- **GPU Acceleration** - When available
- **Service Failover** - Vertex AI → FASHN → Local

### Scaling
- Horizontal: Add more Celery workers
- Vertical: Increase worker concurrency
- Caching: Extend Redis TTL for frequent requests

## Monitoring

### Flower Dashboard
```
http://localhost:5555
```

### Health Check
```http
GET /api/v1/health
```

### Logs
```bash
# View logs
docker-compose logs -f virtualfit-backend
docker-compose logs -f celery-worker
```

## Testing

### Test Webhook
```http
POST /api/v1/webhook/test
{
  "url": "https://your-site.com/webhook",
  "secret": "optional-secret"
}
```

### Sample Integration
```javascript
// E-commerce site integration example
async function tryOnProduct(customerId, productId) {
  const response = await fetch('/api/v1/tryon/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      api_key: 'your-key',
      customer_image_url: `https://cdn.example.com/customers/${customerId}.jpg`,
      garment_image_url: `https://cdn.example.com/products/${productId}.jpg`,
      product_info: {
        name: 'Product Name',
        category: 'TOP',
        color: 'blue'
      },
      webhook_url: 'https://your-site.com/tryon-complete'
    })
  });
  
  const { job_id } = await response.json();
  return job_id;
}
```

## Production Deployment

### Docker Production
```bash
# Build production image
docker build -f Dockerfile.production -t virtualfit-prod .

# Deploy with compose
docker-compose -f docker-compose.yml up -d
```

### Cloud Deployment
- **AWS**: ECS with Application Load Balancer
- **GCP**: Cloud Run with Cloud SQL
- **Azure**: Container Instances with Redis Cache

## Support

### Documentation
- API Docs: http://localhost:8000/docs
- Integration Examples: `/examples` directory
- Troubleshooting: Check logs and Flower dashboard

### Contact
- GitHub Issues: Report bugs and feature requests
- Email: support@virtualfit.com
- Slack: #virtualfit-integration