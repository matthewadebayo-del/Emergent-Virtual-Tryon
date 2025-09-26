# VirtualFit Dual-Mode Deployment Guide

## Overview
Complete deployment guide for VirtualFit's dual-mode virtual try-on system supporting both standalone web application and e-commerce API integration.

## Quick Deployment

### Option 1: Docker Compose (Recommended)
```bash
# Clone repository
git clone https://github.com/matthewadebayo-del/Emergent-Virtual-Tryon.git
cd Emergent-Virtual-Tryon

# Start all services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/api/v1/health
```

### Option 2: Manual Setup
```bash
# Backend services
cd backend
python start_services.py

# Frontend (separate terminal)
cd frontend
yarn start
```

## Production Deployment

### Kubernetes Deployment
```bash
# Build and deploy
./deploy/deploy.sh production v1.0.0

# Monitor deployment
kubectl get pods -n virtualfit-production
kubectl logs -f deployment/virtualfit-api -n virtualfit-production
```

### Cloud Platforms

#### AWS ECS
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name virtualfit-production

# Deploy task definition
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/virtualfit-backend
gcloud run deploy --image gcr.io/PROJECT-ID/virtualfit-backend --platform managed
```

## Testing

### Run Test Suite
```bash
# Install test dependencies
pip install pytest requests redis

# Run all tests
python tests/run_tests.py

# Run specific test categories
pytest tests/test_dual_mode_system.py::TestDualModeSystem -v
pytest tests/test_dual_mode_system.py::TestSDKIntegration -v
```

### Manual Testing Checklist
- [ ] Standalone web app loads at http://localhost:3000
- [ ] API health check passes at http://localhost:8000/api/v1/health
- [ ] Async processing workflow completes successfully
- [ ] Webhook notifications work correctly
- [ ] Redis caching functions properly
- [ ] Celery workers process tasks
- [ ] Flower monitoring accessible at http://localhost:5555

## SDK Integration

### JavaScript SDK
```html
<script src="https://cdn.virtualfit.com/sdk/virtualfit-sdk.js"></script>
<script>
const sdk = new VirtualFitSDK({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.virtualfit.com/api/v1'
});
</script>
```

### Python SDK
```bash
pip install virtualfit-sdk
```

```python
from virtualfit_sdk import VirtualFitSDK

sdk = VirtualFitSDK(api_key='your-api-key')
result = sdk.process_sync(customer_url, garment_url, product_info)
```

## E-commerce Platform Plugins

### Shopify
1. Upload `plugins/shopify/virtualfit-app.js` to theme assets
2. Add to product template: `{% include 'virtualfit-app' %}`
3. Configure API key in theme settings

### WooCommerce
1. Upload `plugins/woocommerce/` to `/wp-content/plugins/`
2. Activate plugin in WordPress admin
3. Configure API settings under Settings > VirtualFit

### Magento
1. Copy plugin files to `app/code/VirtualFit/TryOn/`
2. Run: `php bin/magento setup:upgrade`
3. Configure in Admin > Stores > Configuration > VirtualFit

## Monitoring & Maintenance

### Service Monitoring
- **API Health**: http://localhost:8000/api/v1/health
- **Flower Dashboard**: http://localhost:5555
- **Redis Monitor**: `redis-cli monitor`
- **Logs**: `docker-compose logs -f`

### Performance Optimization
- Scale Celery workers: `docker-compose up --scale celery-worker=5`
- Redis memory optimization: Set appropriate `maxmemory` policy
- Database indexing: Ensure proper MongoDB indexes
- CDN integration: Use CloudFront/CloudFlare for image delivery

### Backup & Recovery
```bash
# Database backup
mongodump --host localhost:27017 --db virtualfit_production

# Redis backup
redis-cli --rdb backup.rdb

# Application backup
tar -czf virtualfit-backup.tar.gz backend/ frontend/ sdks/ plugins/
```

## Troubleshooting

### Common Issues
1. **Services not starting**: Check Docker/Redis/MongoDB status
2. **API timeouts**: Increase worker concurrency or add more workers
3. **Memory issues**: Adjust container resource limits
4. **Webhook failures**: Verify endpoint accessibility and retry logic

### Debug Commands
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs virtualfit-backend
docker-compose logs celery-worker

# Redis connection test
redis-cli ping

# API test
curl -X POST http://localhost:8000/api/v1/webhook/test \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org/post"}'
```

## Security

### API Security
- Use strong API keys (32+ characters)
- Implement rate limiting
- Enable HTTPS in production
- Validate all input parameters
- Use webhook secrets for verification

### Infrastructure Security
- Network isolation with VPC/subnets
- Security groups/firewall rules
- Regular security updates
- SSL/TLS certificates
- Database access controls

## Scaling

### Horizontal Scaling
- Add more API replicas: `kubectl scale deployment virtualfit-api --replicas=5`
- Increase Celery workers: `kubectl scale deployment celery-worker --replicas=10`
- Redis clustering for high availability
- Load balancer configuration

### Vertical Scaling
- Increase container resources
- Optimize worker concurrency
- Database performance tuning
- Cache optimization

## Support

### Documentation
- API Documentation: http://localhost:8000/docs
- Integration Guide: `/INTEGRATION_GUIDE.md`
- SDK Examples: `/examples/` directory

### Monitoring Dashboards
- Grafana: System metrics and performance
- Prometheus: Application metrics collection
- ELK Stack: Centralized logging and analysis

### Contact
- GitHub Issues: Bug reports and feature requests
- Email: support@virtualfit.com
- Slack: #virtualfit-support