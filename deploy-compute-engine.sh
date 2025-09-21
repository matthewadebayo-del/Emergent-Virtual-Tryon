#!/bin/bash

# VirtualFit Compute Engine Deployment with Full 3D Stack
# This script deploys the complete 3D virtual try-on pipeline

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE_NAME="virtualfit-3d-server"
MACHINE_TYPE="n1-standard-8"  # 8 vCPUs, 30GB RAM
DISK_SIZE="100GB"

echo "🚀 Deploying VirtualFit to Compute Engine with Full 3D Stack..."

# Create Compute Engine instance with GPU support
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20231213,mode=rw,size=$DISK_SIZE,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-standard \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=app=virtualfit,env=production \
    --reservation-affinity=any

echo "✅ Compute Engine instance created: $INSTANCE_NAME"

# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
sleep 30

# Copy deployment files to instance
echo "📁 Copying deployment files..."
gcloud compute scp --zone=$ZONE --recurse . $INSTANCE_NAME:~/virtualfit/

# Setup instance with full 3D stack
echo "🔧 Setting up full 3D stack on Compute Engine..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    # Update system
    sudo apt-get update -y
    sudo apt-get upgrade -y
    
    # Install Docker
    sudo apt-get install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker \$USER
    
    # Install Python and build tools
    sudo apt-get install -y python3.9 python3.9-dev python3-pip
    sudo apt-get install -y build-essential cmake pkg-config
    
    # Install system libraries for 3D stack
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
    sudo apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
    sudo apt-get install -y libopencv-dev python3-opencv
    sudo apt-get install -y blender
    
    # Install NVIDIA drivers for GPU support (optional)
    sudo apt-get install -y nvidia-driver-470
    
    cd ~/virtualfit
    
    # Build full 3D Docker image
    sudo docker build -f backend/Dockerfile -t virtualfit-backend-full .
    
    # Start the application
    sudo docker run -d \
        --name virtualfit-backend \
        --restart unless-stopped \
        -p 8000:8000 \
        -e ENABLE_3D_FEATURES=true \
        -e ENABLE_AI_ENHANCEMENT=true \
        -e MONGO_URL=\$MONGO_URL \
        -e OPENAI_API_KEY=\$OPENAI_API_KEY \
        -e FAL_KEY=\$FAL_KEY \
        virtualfit-backend-full
"

# Configure firewall
echo "🔥 Configuring firewall..."
gcloud compute firewall-rules create virtualfit-allow-8000 \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow VirtualFit backend on port 8000"

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "🎉 Deployment Complete!"
echo "📍 Instance: $INSTANCE_NAME"
echo "🌐 External IP: $EXTERNAL_IP"
echo "🔗 Backend URL: http://$EXTERNAL_IP:8000"
echo "📚 API Docs: http://$EXTERNAL_IP:8000/docs"
echo ""
echo "🔧 To check status:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='sudo docker logs virtualfit-backend'"