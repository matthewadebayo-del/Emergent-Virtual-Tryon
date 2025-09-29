#!/bin/bash
echo "Starting virtual display..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
export DISPLAY=:99
sleep 2

# Store original PYTHONPATH
export PYTHONPATH_ORIGINAL="$PYTHONPATH"

echo "Configuring Blender Python paths..."
BLENDER_VERSION=$(blender --version | head -1 | grep -oP "\d+\.\d+" || echo "4.0")
echo "Detected Blender version: $BLENDER_VERSION"

# Initialize PYTHONPATH with Blender scripts
export PYTHONPATH="/usr/share/blender/scripts/modules"

# Add version-specific paths
for path in /usr/share/blender/$BLENDER_VERSION/python/lib/python*/site-packages; do
    if [ -d "$path" ]; then
        export PYTHONPATH="$PYTHONPATH:$path"
        echo "Added version-specific path: $path"
    fi
done

# Add fallback paths for any Blender version
for path in /usr/share/blender/*/python/lib/python*/site-packages; do
    if [ -d "$path" ] && [[ ":$PYTHONPATH:" != *":$path:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$path"
        echo "Added fallback path: $path"
    fi
done

# Add system Python path as fallback
export PYTHONPATH="$PYTHONPATH:$PYTHONPATH_ORIGINAL"
echo "Final PYTHONPATH: $PYTHONPATH"

echo "Testing dependencies..."
python3 -c "
import sys

# Test core 3D dependencies (critical)
try:
    import trimesh, mediapipe, open3d
    print('Core 3D dependencies: OK')
except ImportError as e:
    print(f'Critical 3D dependency missing: {e}')
    sys.exit(1)

# Test Blender subprocess (no bpy import needed)
import subprocess
try:
    result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        version = result.stdout.split('\n')[0] if result.stdout else 'Unknown'
        print(f'Blender subprocess: OK - {version}')
    else:
        print('Blender subprocess: Failed - using fallback rendering')
except Exception as e:
    print(f'Blender subprocess unavailable: {e} - using fallback rendering')

# Test AI dependencies (non-critical)
try:
    import torch, transformers, diffusers
    print('AI dependencies: OK')
except ImportError as e:
    print(f'AI dependencies unavailable: {e}')
    print('AI enhancement will be disabled')
    import os
    os.environ['DISABLE_AI_ENHANCEMENT'] = 'true'

print('Startup checks complete')
"

echo "Starting application..."
uvicorn production_server:app --host 0.0.0.0 --port $PORT