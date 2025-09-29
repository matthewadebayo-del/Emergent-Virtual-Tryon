#!/bin/bash
# Quick server start script

cd backend

# Kill existing server
pkill -f "production_server.py" || echo "No existing server"

# Start server in background
echo "Starting VirtualFit backend server..."
nohup python production_server.py > server.log 2>&1 &

echo "Server started! Check logs: tail -f backend/server.log"
echo "API available at: http://localhost:8000"