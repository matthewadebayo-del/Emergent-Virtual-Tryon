#!/bin/bash
# Quick frontend start script

cd frontend

# Kill existing frontend
pkill -f "react-scripts start" || echo "No existing frontend"

# Start frontend in background
echo "Starting VirtualFit frontend..."
nohup yarn start > frontend.log 2>&1 &

echo "Frontend started! Check logs: tail -f frontend/frontend.log"
echo "Frontend available at: http://localhost:3000"