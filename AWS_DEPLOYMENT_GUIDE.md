# AWS Deployment Guide for VirtualFit Virtual Try-On Application

## Overview
This guide covers deploying your VirtualFit application on AWS with custom domain, SSL certificates, and production-grade infrastructure.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CloudFront    │────│   Application    │────│   DocumentDB    │
│   (CDN + SSL)   │    │  Load Balancer   │    │   (MongoDB)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────────────┐
                       │  Elastic       │
                       │  Beanstalk     │
                       │  (Backend)     │
                       └────────────────┘
```

## Prerequisites
- AWS Account with CLI configured
- Domain name registered
- GitHub repository with your VirtualFit code

## Step 1: Prepare Your Application

### 1.1 Environment Configuration
Create production environment files:

```bash
# backend/.env.production
MONGO_URL="mongodb://your-documentdb-cluster:27017"
DB_NAME="virtualfit_production"
CORS_ORIGINS="https://yourdomain.com"
OPENAI_API_KEY="your-openai-key"
JWT_SECRET="your-secure-jwt-secret"
```

```bash
# frontend/.env.production
REACT_APP_BACKEND_URL=https://api.yourdomain.com
```

### 1.2 Docker Configuration
Create `Dockerfile` for backend:

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=${MONGO_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=${REACT_APP_BACKEND_URL}
```

## Step 2: Database Setup (Amazon DocumentDB)

### 2.1 Create DocumentDB Cluster

```bash
# Create DocumentDB cluster
aws docdb create-db-cluster \
  --db-cluster-identifier virtualfit-cluster \
  --engine docdb \
  --master-username admin \
  --master-user-password SecurePassword123 \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name default
```

### 2.2 Create DocumentDB Instance

```bash
# Create DocumentDB instance
aws docdb create-db-instance \
  --db-instance-identifier virtualfit-instance \
  --db-instance-class db.t3.medium \
  --engine docdb \
  --db-cluster-identifier virtualfit-cluster
```

## Step 3: Backend Deployment (Elastic Beanstalk)

### 3.1 Install EB CLI

```bash
pip install awsebcli
```

### 3.2 Initialize Elastic Beanstalk

```bash
cd backend
eb init -p python-3.11 virtualfit-backend
```

### 3.3 Create Environment

```bash
eb create virtualfit-production \
  --instance-type t3.medium \
  --min-instances 2 \
  --max-instances 10
```

### 3.4 Configure Environment Variables

```bash
eb setenv \
  MONGO_URL="mongodb://admin:SecurePassword123@virtualfit-cluster.cluster-xxx.docdb.amazonaws.com:27017" \
  OPENAI_API_KEY="your-openai-key" \
  JWT_SECRET="your-jwt-secret"
```

### 3.5 Deploy Backend

```bash
eb deploy
```

## Step 4: Frontend Deployment (S3 + CloudFront)

### 4.1 Build React Application

```bash
cd frontend
npm run build
```

### 4.2 Create S3 Bucket

```bash
aws s3 mb s3://virtualfit-frontend-bucket
```

### 4.3 Upload Frontend Files

```bash
aws s3 sync build/ s3://virtualfit-frontend-bucket --delete
```

### 4.4 Configure S3 for Static Hosting

```bash
aws s3 website s3://virtualfit-frontend-bucket \
  --index-document index.html \
  --error-document index.html
```

### 4.5 Create CloudFront Distribution

```json
{
  "CallerReference": "virtualfit-distribution",
  "Comment": "VirtualFit Frontend Distribution",
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "S3-virtualfit-frontend-bucket",
        "DomainName": "virtualfit-frontend-bucket.s3.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-virtualfit-frontend-bucket",
    "ViewerProtocolPolicy": "redirect-to-https",
    "Compress": true
  }
}
```

## Step 5: SSL Certificate & Custom Domain

### 5.1 Request SSL Certificate

```bash
aws acm request-certificate \
  --domain-name yourdomain.com \
  --subject-alternative-names "*.yourdomain.com" \
  --validation-method DNS
```

### 5.2 Update Route 53 DNS

```bash
# Create hosted zone
aws route53 create-hosted-zone \
  --name yourdomain.com \
  --caller-reference $(date +%s)

# Add A records for your domain
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_ZONE_ID \
  --change-batch file://dns-records.json
```

DNS Records file (`dns-records.json`):
```json
{
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "DNSName": "your-cloudfront-domain.cloudfront.net",
          "EvaluateTargetHealth": false,
          "HostedZoneId": "Z2FDTNDATAQYW2"
        }
      }
    },
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.yourdomain.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [
          {
            "Value": "virtualfit-production.us-east-1.elasticbeanstalk.com"
          }
        ]
      }
    }
  ]
}
```

## Step 6: Load Balancer & Auto Scaling

### 6.1 Configure Application Load Balancer

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name virtualfit-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-abcdef
```

### 6.2 Configure Target Groups

```bash
# Create target group
aws elbv2 create-target-group \
  --name virtualfit-targets \
  --protocol HTTP \
  --port 8001 \
  --vpc-id vpc-12345678
```

## Step 7: Monitoring & Logging

### 7.1 CloudWatch Setup

```bash
# Create log group
aws logs create-log-group \
  --log-group-name /aws/elasticbeanstalk/virtualfit-production
```

### 7.2 Set Up Alarms

```bash
# CPU utilization alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "VirtualFit-HighCPU" \
  --alarm-description "Alarm when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80.0 \
  --comparison-operator GreaterThanThreshold
```

## Step 8: CI/CD Pipeline (Optional)

### 8.1 Create GitHub Actions Workflow

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to Elastic Beanstalk
      run: |
        cd backend
        eb deploy virtualfit-production
    
    - name: Build and Deploy Frontend
      run: |
        cd frontend
        npm ci
        npm run build
        aws s3 sync build/ s3://virtualfit-frontend-bucket --delete
        aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

## Step 9: Security Configuration

### 9.1 WAF Setup

```bash
# Create WAF web ACL
aws wafv2 create-web-acl \
  --name VirtualFitWAF \
  --scope CLOUDFRONT \
  --default-action Allow={} \
  --rules file://waf-rules.json
```

### 9.2 Security Groups

```bash
# Backend security group
aws ec2 create-security-group \
  --group-name virtualfit-backend-sg \
  --description "VirtualFit Backend Security Group"

# Allow HTTPS from anywhere
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

## Step 10: Cost Optimization

### 10.1 Reserved Instances

```bash
# Purchase reserved instances for predictable workloads
aws ec2 purchase-reserved-instances-offering \
  --reserved-instances-offering-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --instance-count 2
```

### 10.2 Auto Scaling Policies

```bash
# Create scaling policy
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name virtualfit-asg \
  --policy-name virtualfit-scale-up \
  --scaling-adjustment 2 \
  --adjustment-type ChangeInCapacity
```

## Deployment Commands Summary

```bash
# Complete deployment script
#!/bin/bash

# 1. Build and deploy backend
cd backend
eb deploy virtualfit-production

# 2. Build and deploy frontend
cd ../frontend
npm run build
aws s3 sync build/ s3://virtualfit-frontend-bucket --delete

# 3. Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id YOUR_DISTRIBUTION_ID \
  --paths "/*"

# 4. Update database if needed
# (Run any migration scripts)

echo "Deployment completed successfully!"
```

## Estimated Monthly Costs

- **EC2 Instances (t3.medium x2)**: ~$60
- **DocumentDB (db.t3.medium)**: ~$50
- **CloudFront**: ~$10-50 (depending on traffic)
- **S3 Storage**: ~$5
- **Load Balancer**: ~$20
- **Route 53**: ~$1
- **Total**: ~$150-200/month

## Maintenance Tasks

### Daily
- Monitor CloudWatch dashboards
- Check error logs

### Weekly
- Review costs and optimize
- Update security patches

### Monthly
- Review scaling metrics
- Update SSL certificates if needed
- Backup database

## Troubleshooting

### Common Issues:
1. **CORS Errors**: Check CORS_ORIGINS environment variable
2. **Database Connection**: Verify DocumentDB security groups
3. **SSL Issues**: Ensure certificate is validated and attached
4. **High Latency**: Check CloudFront cache settings

### Debug Commands:
```bash
# Check Elastic Beanstalk logs
eb logs

# View CloudWatch logs
aws logs describe-log-streams --log-group-name /aws/elasticbeanstalk/virtualfit-production

# Test backend health
curl https://api.yourdomain.com/api/

# Test frontend
curl https://yourdomain.com
```

This completes your AWS deployment setup for the VirtualFit application with enterprise-grade infrastructure, monitoring, and security.
