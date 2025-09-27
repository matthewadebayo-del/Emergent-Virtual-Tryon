#!/bin/bash

# Quick Update Script - Pull latest code and restart service
# Use this after pushing changes to GitHub

set -e

echo "🔄 Quick update from GitHub..."

cd /opt/virtualfit
git pull origin main

echo "🐍 Activating environment and updating dependencies..."
source venv/bin/activate
pip install -r backend/requirements.txt

echo "🔧 Restarting service..."
sudo systemctl restart virtualfit

echo "⏳ Waiting for service to start..."
sleep 5

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Update successful - service is running"
else
    echo "❌ Service failed to start, checking logs..."
    sudo journalctl -u virtualfit --no-pager -n 10
fi