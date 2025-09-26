#!/bin/bash
# Production deployment script for VirtualFit dual-mode system

set -e

echo "🚀 VirtualFit Production Deployment"
echo "=================================="

# Configuration
DOCKER_REGISTRY="virtualfit"
VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"production"}

# Build and push images
echo "📦 Building Docker images..."
docker build -f backend/Dockerfile.production -t $DOCKER_REGISTRY/backend:$VERSION backend/
docker build -f frontend/Dockerfile -t $DOCKER_REGISTRY/frontend:$VERSION frontend/

echo "📤 Pushing to registry..."
docker push $DOCKER_REGISTRY/backend:$VERSION
docker push $DOCKER_REGISTRY/frontend:$VERSION

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f deploy/production-deploy.yml

# Wait for deployment
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/virtualfit-api -n virtualfit-production
kubectl rollout status deployment/celery-worker -n virtualfit-production

# Get service URL
echo "🌐 Getting service URL..."
SERVICE_URL=$(kubectl get service virtualfit-api-service -n virtualfit-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "✅ Deployment complete!"
echo "📊 Service URLs:"
echo "  • API: http://$SERVICE_URL"
echo "  • Health: http://$SERVICE_URL/api/v1/health"
echo "  • Docs: http://$SERVICE_URL/docs"

# Run health check
echo "🔍 Running health check..."
sleep 30
curl -f http://$SERVICE_URL/api/v1/health || echo "❌ Health check failed"

echo "🎉 VirtualFit is live!"