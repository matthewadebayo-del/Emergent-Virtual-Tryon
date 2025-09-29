# Server Management Commands

## Deploy Latest Changes
```bash
# Pull from GitHub and restart server
./deploy_production.sh
```

## Quick Server Management
```bash
# Start server
./start_server.sh

# Stop server
pkill -f "production_server.py"

# Check server status
ps aux | grep production_server.py

# View logs
tail -f backend/server.log

# Check if server is responding
curl http://localhost:8000/health
```

## Manual Deployment Steps
```bash
# 1. Pull latest code
git pull origin main

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Start server in background
nohup python production_server.py > server.log 2>&1 &

# 4. Verify server is running
curl http://localhost:8000/
```

## Server Endpoints
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Main API**: http://localhost:8000/api/
- **Virtual Try-On**: http://localhost:8000/api/virtual-tryon

## Environment Setup
Make sure `.env` file contains:
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=virtualfit_production
SECRET_KEY=virtualfit-production-secret-key-change-this
FASHN_API_KEY=fa-hfuTx6vji1ma-KeeQTDToyTwCObNZrPbtnO9w
```