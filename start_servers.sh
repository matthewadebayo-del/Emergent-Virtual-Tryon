#!/bin/bash
set -e

echo "Starting Virtual Try-On servers with monitoring..."

cd backend
python -c "import fastapi, uvicorn, firebase_admin; print('✅ Core dependencies available')"

echo "Starting backend server..."
uvicorn server:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

sleep 5
curl -f http://localhost:8001/api/health || (echo "❌ Backend failed to start" && exit 1)

cd ../frontend
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!

cd ..
python monitor_servers.py &
MONITOR_PID=$!

echo "✅ All servers started successfully"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Monitor PID: $MONITOR_PID"
