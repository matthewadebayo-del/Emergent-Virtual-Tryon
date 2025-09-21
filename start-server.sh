#!/bin/bash

# Start VirtualFit backend server on Compute Engine
echo "🚀 Starting VirtualFit backend server..."

# Navigate to project directory
cd ~/virtualfit

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
export MONGO_URL="mongodb://localhost:27017"
export DB_NAME="virtualfit_production"
export CORS_ORIGINS="http://localhost:3000,https://virtual-tryon-app-a8pe83vz.devinapps.com,*"
export OPENAI_API_KEY="your-openai-api-key-here"

# Install any missing dependencies
echo "📦 Installing missing dependencies..."
pip install openai python-jose[cryptography] python-multipart motor bcrypt

# Start the server using the custom runner
echo "🎯 Starting server on port 8000..."
cd backend
python3 run_server.py