# Google Cloud Platform & Firebase Deployment Guide

## Overview
Deploy your VirtualFit application on Google Cloud Platform with Firebase integration for optimal performance and scalability.

## Architecture Options

### Option 1: Full GCP Stack
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloud CDN     │────│   Cloud Run      │────│   Cloud         │
│   + Load        │    │   (Backend)      │    │   Firestore     │
│   Balancer      │    └──────────────────┘    │   (Database)    │
└─────────────────┘    ┌──────────────────┐    └─────────────────┘
                       │   Firebase       │
                       │   Hosting        │
                       │   (Frontend)     │
                       └──────────────────┘
```

### Option 2: Firebase-First Stack
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Firebase      │────│   Cloud          │────│   Firestore     │
│   Hosting       │    │   Functions      │    │   (Database)    │
│   (Frontend)    │    │   (Backend)      │    └─────────────────┘
└─────────────────┘    └──────────────────┘
```

## Prerequisites
- Google Cloud Account with billing enabled
- Firebase project created
- Domain name registered
- GitHub repository with your code

---

## Option 1: Full Google Cloud Platform Deployment

### Step 1: Setup GCP Environment

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Set project
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  firestore.googleapis.com \
  cloudresourcemanager.googleapis.com
```

### Step 2: Database Setup (Firestore)

```bash
# Initialize Firestore
gcloud firestore databases create --region=us-central1
```

Update backend to use Firestore:

```python
# backend/firestore_config.py
from google.cloud import firestore
import os

db = firestore.Client(project=os.environ.get('GCP_PROJECT_ID'))

class FirestoreModels:
    @staticmethod
    async def create_user(user_data):
        doc_ref = db.collection('users').document(user_data['id'])
        doc_ref.set(user_data)
        return user_data
    
    @staticmethod
    async def get_user(user_id):
        doc_ref = db.collection('users').document(user_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    
    @staticmethod
    async def save_measurements(user_id, measurements):
        doc_ref = db.collection('users').document(user_id)
        doc_ref.update({'measurements': measurements})
```

### Step 3: Backend Deployment (Cloud Run)

Create `Dockerfile`:

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

Update `requirements.txt`:

```text
# Add to existing requirements.txt
google-cloud-firestore>=2.11.0
google-auth>=2.17.0
```

Deploy to Cloud Run:

```bash
# Build and deploy
cd backend
gcloud run deploy virtualfit-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="OPENAI_API_KEY=your-key,GCP_PROJECT_ID=your-project-id"
```

### Step 4: Frontend Deployment (Firebase Hosting)

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Initialize Firebase in frontend directory
cd frontend
firebase init hosting

# Select your Firebase project
# Set public directory to 'build'
# Configure as single-page app: Yes
```

Update `firebase.json`:

```json
{
  "hosting": {
    "public": "build",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ],
    "headers": [
      {
        "source": "/static/**",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000"
          }
        ]
      }
    ]
  }
}
```

Build and deploy:

```bash
# Build React app
npm run build

# Deploy to Firebase
firebase deploy --only hosting
```

### Step 5: Custom Domain Setup

```bash
# Add custom domain to Firebase
firebase hosting:sites:create your-domain-name

# Connect domain
firebase target:apply hosting production your-domain-name
firebase hosting:channel:deploy live --only hosting:production
```

Update DNS records:
- Add A record: `151.101.1.195` and `151.101.65.195`
- Add AAAA record: `2a04:4e42::703` and `2a04:4e42:200::703`

---

## Option 2: Firebase-First Deployment

### Step 1: Setup Firebase Environment

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize Firebase project
firebase init

# Select:
# - Hosting
# - Functions
# - Firestore
```

### Step 2: Convert Backend to Cloud Functions

Create `functions/package.json`:

```json
{
  "name": "virtualfit-functions",
  "scripts": {
    "serve": "firebase emulators:start --only functions",
    "shell": "firebase functions:shell",
    "start": "npm run shell",
    "deploy": "firebase deploy --only functions",
    "logs": "firebase functions:log"
  },
  "engines": {
    "node": "18"
  },
  "dependencies": {
    "firebase-admin": "^11.8.0",
    "firebase-functions": "^4.3.1",
    "express": "^4.18.2",
    "cors": "^2.8.5"
  }
}
```

Create `functions/index.js`:

```javascript
const functions = require('firebase-functions');
const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.firestore();

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: '10mb' }));

// User registration
app.post('/register', async (req, res) => {
  try {
    const { email, password, full_name } = req.body;
    
    // Create user in Firebase Auth
    const userRecord = await admin.auth().createUser({
      email,
      password,
      displayName: full_name
    });
    
    // Save user profile to Firestore
    await db.collection('users').doc(userRecord.uid).set({
      id: userRecord.uid,
      email,
      full_name,
      created_at: admin.firestore.FieldValue.serverTimestamp()
    });
    
    // Create custom token
    const token = await admin.auth().createCustomToken(userRecord.uid);
    
    res.json({ access_token: token, token_type: 'bearer' });
  } catch (error) {
    res.status(400).json({ detail: error.message });
  }
});

// Virtual try-on
app.post('/tryon', async (req, res) => {
  try {
    // Verify authentication
    const token = req.headers.authorization?.split(' ')[1];
    const decodedToken = await admin.auth().verifyIdToken(token);
    
    const { user_image_base64, product_id } = req.body;
    
    // Call OpenAI API using standard OpenAI package
    const { OpenAI } = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    
    const images = await imageGen.generate_images({
      prompt: "Virtual try-on image generation",
      model: "gpt-image-1"
    });
    
    const result = {
      result_image_base64: Buffer.from(images[0]).toString('base64'),
      size_recommendation: "M"
    };
    
    // Save try-on result
    await db.collection('tryon_results').add({
      user_id: decodedToken.uid,
      ...result,
      created_at: admin.firestore.FieldValue.serverTimestamp()
    });
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
});

// Export the API
exports.api = functions.https.onRequest(app);
```

### Step 3: Firebase Configuration

Update `firebase.json`:

```json
{
  "functions": {
    "source": "functions",
    "runtime": "nodejs18"
  },
  "hosting": {
    "public": "build",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "/api/**",
        "function": "api"
      },
      {
        "source": "**",
        "destination": "/index.html"
      }
    ]
  },
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  }
}
```

Create `firestore.rules`:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can read/write their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Try-on results
    match /tryon_results/{resultId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.user_id;
    }
    
    // Products are publicly readable
    match /products/{productId} {
      allow read: if true;
      allow write: if request.auth != null; // Admin only in production
    }
  }
}
```

### Step 4: Environment Configuration

```bash
# Set Firebase environment variables
firebase functions:config:set \
  openai.api_key="your-openai-key" \
  app.cors_origins="https://yourdomain.com"
```

### Step 5: Deploy Firebase Application

```bash
# Deploy functions and hosting
firebase deploy

# Custom domain setup
firebase hosting:sites:create yourdomain
firebase target:apply hosting production yourdomain
firebase hosting:channel:deploy live --only hosting:production
```

---

## Advanced Configuration

### Load Balancing & CDN

```bash
# Create Cloud Load Balancer
gcloud compute url-maps create virtualfit-map \
  --default-service virtualfit-backend

# Create HTTPS proxy
gcloud compute target-https-proxies create virtualfit-https-proxy \
  --url-map virtualfit-map \
  --ssl-certificates virtualfit-ssl-cert

# Create forwarding rule
gcloud compute forwarding-rules create virtualfit-https-rule \
  --global \
  --target-https-proxy virtualfit-https-proxy \
  --ports 443
```

### SSL Certificate

```bash
# Create managed SSL certificate
gcloud compute ssl-certificates create virtualfit-ssl-cert \
  --domains yourdomain.com,www.yourdomain.com \
  --global
```

### Monitoring & Logging

Enable monitoring:

```bash
# Create notification channel
gcloud alpha monitoring channels create \
  --display-name="VirtualFit Email" \
  --type=email \
  --channel-labels=email_address=admin@yourdomain.com
```

### CI/CD with Cloud Build

Create `cloudbuild.yaml`:

```yaml
steps:
  # Build backend
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args: ['run', 'deploy', 'virtualfit-backend', '--source', './backend', '--region', 'us-central1']
  
  # Build and deploy frontend
  - name: 'node:18'
    entrypoint: 'npm'
    args: ['ci']
    dir: 'frontend'
  
  - name: 'node:18'
    entrypoint: 'npm'
    args: ['run', 'build']
    dir: 'frontend'
  
  - name: 'gcr.io/$PROJECT_ID/firebase'
    args: ['deploy', '--only', 'hosting']
    dir: 'frontend'

options:
  logging: CLOUD_LOGGING_ONLY
```

---

## Cost Estimation

### Firebase Option (Monthly)
- **Firebase Hosting**: $0-25 (based on bandwidth)
- **Cloud Functions**: $0-50 (based on invocations)
- **Firestore**: $0-30 (based on reads/writes)
- **Firebase Auth**: $0 (up to 50k users)
- **Total**: $0-105/month

### Full GCP Option (Monthly)
- **Cloud Run**: $10-50 (based on requests)
- **Firebase Hosting**: $0-25
- **Firestore**: $0-30
- **Cloud CDN**: $5-20
- **Load Balancer**: $18
- **Total**: $33-143/month

---

## Deployment Scripts

### Firebase Deployment Script

```bash
#!/bin/bash
# deploy-firebase.sh

echo "🚀 Deploying VirtualFit to Firebase..."

# Build frontend
cd frontend
npm run build

# Deploy functions and hosting
firebase deploy

# Custom domain (if configured)
firebase hosting:channel:deploy live --only hosting:production

echo "✅ Deployment completed!"
echo "🌐 Your app is live at: https://yourdomain.com"
```

### GCP Deployment Script

```bash
#!/bin/bash
# deploy-gcp.sh

echo "🚀 Deploying VirtualFit to Google Cloud..."

# Deploy backend to Cloud Run
cd backend
gcloud run deploy virtualfit-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Deploy frontend to Firebase
cd ../frontend
npm run build
firebase deploy --only hosting

echo "✅ Deployment completed!"
echo "🌐 Backend: https://virtualfit-backend-xxx-uc.a.run.app"
echo "🌐 Frontend: https://yourdomain.com"
```

## Maintenance & Monitoring

### Health Checks

```javascript
// Add to Cloud Functions
exports.healthCheck = functions.https.onRequest((req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0'
  });
});
```

### Backup Strategy

```bash
# Automated Firestore backup
gcloud firestore export gs://your-backup-bucket \
  --collection-ids=users,products,tryon_results
```

This completes your Google Cloud Platform and Firebase deployment options with enterprise-grade scalability and monitoring.
