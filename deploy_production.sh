#!/bin/bash
# Production Deployment Script for VirtualFit Backend
# Run this on the compute server to deploy latest changes

set -e

echo "🚀 VirtualFit Production Deployment"
echo "=================================="

# Update from GitHub
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Navigate to backend directory
cd backend

# Install/update Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Set up environment variables
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "No .env.example found, using current .env"
fi

# Kill existing server process
echo "🛑 Stopping existing server..."
pkill -f "production_server.py" || echo "No existing server found"

# Start server in background
echo "🚀 Starting production server..."
nohup python production_server.py > server.log 2>&1 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "📊 Server running at: http://localhost:8000"
    echo "📋 API docs at: http://localhost:8000/docs"
    echo "📄 Logs: tail -f backend/server.log"
else
    echo "❌ Server failed to start"
    echo "📄 Check logs: cat backend/server.log"
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "💡 To monitor: tail -f backend/server.log"
echo "🛑 To stop: pkill -f production_server.py"