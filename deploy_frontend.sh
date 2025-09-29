#!/bin/bash
# Frontend Deployment Script for VirtualFit

set -e

echo "🎨 VirtualFit Frontend Deployment"
echo "================================="

# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
yarn install

# Create production .env file
echo "⚙️ Setting up environment..."
cat > .env << EOF
REACT_APP_BACKEND_URL=http://localhost:8000
WDS_SOCKET_PORT=443
GENERATE_SOURCEMAP=false
EOF

# Kill existing frontend process
echo "🛑 Stopping existing frontend..."
pkill -f "react-scripts start" || echo "No existing frontend found"

# Start frontend in background
echo "🚀 Starting frontend server..."
nohup yarn start > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

# Check if frontend is running
if ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ Frontend started successfully (PID: $FRONTEND_PID)"
    echo "🌐 Frontend running at: http://localhost:3000"
    echo "📄 Logs: tail -f frontend/frontend.log"
else
    echo "❌ Frontend failed to start"
    echo "📄 Check logs: cat frontend/frontend.log"
    exit 1
fi

echo ""
echo "🎉 Frontend deployment completed!"
echo "💡 To monitor: tail -f frontend/frontend.log"
echo "🛑 To stop: pkill -f 'react-scripts start'"