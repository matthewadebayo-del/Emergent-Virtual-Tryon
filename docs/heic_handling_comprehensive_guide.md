# HEIC Handling in Virtual Try-On Applications

## Overview
HEIC (High Efficiency Image Container) is Apple's modern image format that provides better compression than JPEG while maintaining quality. However, it requires special handling in web applications due to limited browser support.

## Browser Support Issues
- **Safari**: Native HEIC support
- **Chrome/Firefox/Edge**: No native HEIC support
- **Mobile browsers**: Limited support outside iOS Safari

## Frontend HEIC Processing Strategy

### 1. Client-Side Detection and Conversion

```javascript
// VirtualTryOn.js - Enhanced HEIC handling
const handleImageUpload = (event, type) => {
  const file = event.target.files[0];
  if (!file) return;

  console.log('File selected:', file.name, 'Type:', file.type, 'Size:', file.size);
  
  // Check if file is HEIC
  const isHeic = file.type === 'image/heic' || 
                 file.type === 'image/heif' || 
                 file.name.toLowerCase().endsWith('.heic') ||
                 file.name.toLowerCase().endsWith('.heif');

  if (isHeic) {
    console.log('HEIC file detected, processing...');
    handleHeicFile(file, type);
  } else {
    handleStandardImageFile(file, type);
  }
};

const handleHeicFile = async (file, type) => {
  try {
    // Method 1: Try heic2any library (if available)
    if (window.heic2any) {
      console.log('Converting HEIC using heic2any...');
      const convertedBlob = await heic2any({
        blob: file,
        toType: "image/jpeg",
        quality: 0.9
      });
      
      const convertedFile = new File([convertedBlob], 
        file.name.replace(/\.(heic|heif)$/i, '.jpg'), 
        { type: 'image/jpeg' }
      );
      
      handleStandardImageFile(convertedFile, type);
      return;
    }

    // Method 2: Send to backend for conversion
    console.log('Sending HEIC to backend for conversion...');
    const formData = new FormData();
    formData.append('heic_file', file);
    
    const response = await fetch('/api/convert-heic', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: formData
    });
    
    if (response.ok) {
      const convertedBlob = await response.blob();
      const convertedFile = new File([convertedBlob], 
        file.name.replace(/\.(heic|heif)$/i, '.jpg'), 
        { type: 'image/jpeg' }
      );
      handleStandardImageFile(convertedFile, type);
    } else {
      throw new Error('Backend HEIC conversion failed');
    }
    
  } catch (error) {
    console.error('HEIC processing failed:', error);
    alert('HEIC file could not be processed. Please convert to JPG/PNG or try a different image.');
  }
};

const handleStandardImageFile = (file, type) => {
  const reader = new FileReader();
  
  reader.onload = (e) => {
    const base64 = e.target.result;
    console.log('File read successfully, base64 length:', base64.length);
    
    // Validate base64 data format
    if (!base64 || !base64.startsWith('data:image/')) {
      console.error('Invalid image data format');
      alert('Failed to process the selected file. Please try a different image.');
      return;
    }
    
    if (type === 'user') {
      setUserImage(base64);
      setUserImagePreview(base64);
      console.log('User image set successfully');
    } else if (type === 'clothing') {
      setClothingImage(base64);
      setClothingImagePreview(base64);
      console.log('Clothing image set successfully');
    }
  };
  
  reader.onerror = (error) => {
    console.error('File reading failed:', error);
    alert('Failed to read the selected file. Please try again.');
  };
  
  reader.readAsDataURL(file);
};
```

### 2. Include HEIC Conversion Library

```html
<!-- Add to public/index.html -->
<script src="https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js"></script>
```

## Backend HEIC Processing

### 1. Python HEIC Processor

```python
# backend/src/utils/heic_processor.py
import io
import logging
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class HEICProcessor:
    """Process HEIC images with multiple fallback methods"""
    
    def __init__(self):
        self.conversion_methods = [
            self._convert_with_pillow_heif,
            self._convert_with_pyheif,
            self._convert_with_imageio
        ]
    
    def convert_heic_to_jpeg(self, heic_data: bytes, quality: int = 90) -> Optional[bytes]:
        """Convert HEIC data to JPEG with multiple fallback methods"""
        
        for i, method in enumerate(self.conversion_methods):
            try:
                logger.info(f"Attempting HEIC conversion method {i+1}")
                jpeg_data = method(heic_data, quality)
                if jpeg_data:
                    logger.info(f"✅ HEIC conversion successful with method {i+1}")
                    return jpeg_data
            except Exception as e:
                logger.warning(f"⚠️ HEIC conversion method {i+1} failed: {e}")
                continue
        
        logger.error("❌ All HEIC conversion methods failed")
        return None
    
    def _convert_with_pillow_heif(self, heic_data: bytes, quality: int) -> Optional[bytes]:
        """Convert using pillow-heif (recommended)"""
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            
            # Open HEIC image
            image = Image.open(io.BytesIO(heic_data))
            
            # Fix orientation if needed
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(0x0112)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Save as JPEG
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
            
            return output_buffer.getvalue()
            
        except ImportError:
            raise Exception("pillow-heif not available")
    
    def _convert_with_pyheif(self, heic_data: bytes, quality: int) -> Optional[bytes]:
        """Convert using pyheif (alternative)"""
        try:
            import pyheif
            
            # Decode HEIC
            heif_file = pyheif.read(heic_data)
            
            # Convert to PIL Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            
            # Apply rotation if needed
            if "exif" in heif_file.metadata:
                # Handle EXIF orientation
                pass  # Implement EXIF orientation handling
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=quality)
            
            return output_buffer.getvalue()
            
        except ImportError:
            raise Exception("pyheif not available")
    
    def _convert_with_imageio(self, heic_data: bytes, quality: int) -> Optional[bytes]:
        """Convert using imageio (fallback)"""
        try:
            import imageio
            import numpy as np
            
            # Read HEIC data
            image_array = imageio.imread(io.BytesIO(heic_data))
            
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=quality)
            
            return output_buffer.getvalue()
            
        except ImportError:
            raise Exception("imageio not available")

# Global processor instance
heic_processor = HEICProcessor()
```

### 2. FastAPI Endpoint for HEIC Conversion

```python
# backend/server.py - Add HEIC conversion endpoint
from src.utils.heic_processor import heic_processor

@app.post("/api/convert-heic")
async def convert_heic_endpoint(
    heic_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Convert HEIC file to JPEG"""
    try:
        # Read HEIC file data
        heic_data = await heic_file.read()
        logger.info(f"Received HEIC file: {heic_file.filename}, size: {len(heic_data)} bytes")
        
        # Convert to JPEG
        jpeg_data = heic_processor.convert_heic_to_jpeg(heic_data, quality=90)
        
        if jpeg_data:
            logger.info(f"✅ HEIC conversion successful, output size: {len(jpeg_data)} bytes")
            
            # Return JPEG data
            return Response(
                content=jpeg_data,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f"attachment; filename={heic_file.filename.replace('.heic', '.jpg').replace('.heif', '.jpg')}"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="HEIC conversion failed")
            
    except Exception as e:
        logger.error(f"❌ HEIC conversion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"HEIC processing failed: {str(e)}")

# Enhanced image upload handling
def convert_heic_to_jpeg(image_data: bytes) -> bytes:
    """Convert HEIC image data to JPEG if needed"""
    try:
        # Check if data is HEIC (basic check)
        if image_data.startswith(b'\x00\x00\x00\x20ftypheic') or image_data.startswith(b'\x00\x00\x00\x18ftypheic'):
            logger.info("HEIC format detected, converting...")
            
            jpeg_data = heic_processor.convert_heic_to_jpeg(image_data)
            if jpeg_data:
                logger.info("✅ HEIC converted to JPEG successfully")
                return jpeg_data
            else:
                logger.warning("⚠️ HEIC conversion failed, returning original data")
                return image_data
        else:
            # Not HEIC, return as-is
            return image_data
            
    except Exception as e:
        logger.error(f"❌ HEIC conversion error: {e}")
        return image_data
```

### 3. Requirements for HEIC Support

```txt
# Add to requirements.txt
pillow-heif>=0.10.0  # Primary HEIC support
pyheif>=0.7.1        # Alternative HEIC support
imageio>=2.25.0      # Fallback image processing
```

## Error Handling and User Experience

### 1. Progressive Enhancement

```javascript
// Check for HEIC support and show appropriate UI
const checkHeicSupport = () => {
  // Check if browser supports HEIC natively
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();
  
  return new Promise((resolve) => {
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = 'data:image/heic;base64,'; // Minimal HEIC data
  });
};

// Show appropriate file input accept attribute
const getAcceptedFormats = async () => {
  const heicSupported = await checkHeicSupport();
  return heicSupported 
    ? 'image/*,.heic,.heif' 
    : 'image/jpeg,image/png,image/gif,image/webp';
};
```

### 2. User-Friendly Error Messages

```javascript
const showHeicError = (error) => {
  const errorMessages = {
    'conversion_failed': 'Your HEIC image could not be processed. Please try converting it to JPG first.',
    'file_too_large': 'Your HEIC file is too large. Please try a smaller image or convert to JPG.',
    'unsupported_format': 'HEIC files are not fully supported. Please use JPG, PNG, or other standard formats.',
    'network_error': 'Could not process your image due to a network error. Please try again.'
  };
  
  const message = errorMessages[error.type] || 'An error occurred processing your image.';
  
  // Show user-friendly error dialog
  showErrorDialog({
    title: 'Image Processing Error',
    message: message,
    actions: [
      { text: 'Try Again', action: () => retryImageUpload() },
      { text: 'Choose Different Image', action: () => openFileSelector() }
    ]
  });
};
```

## Testing HEIC Functionality

### 1. Test Cases

```javascript
// Test different HEIC scenarios
const testHeicHandling = async () => {
  const testCases = [
    { name: 'Standard HEIC', file: 'test.heic' },
    { name: 'HEIF variant', file: 'test.heif' },
    { name: 'Large HEIC', file: 'large.heic' },
    { name: 'Rotated HEIC', file: 'rotated.heic' },
    { name: 'HEIC with transparency', file: 'transparent.heic' }
  ];
  
  for (const testCase of testCases) {
    try {
      console.log(`Testing: ${testCase.name}`);
      const result = await processHeicFile(testCase.file);
      console.log(`✅ ${testCase.name}: Success`);
    } catch (error) {
      console.error(`❌ ${testCase.name}: ${error.message}`);
    }
  }
};
```

### 2. Performance Monitoring

```python
# Monitor HEIC conversion performance
import time
from functools import wraps

def monitor_heic_conversion(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"✅ HEIC conversion completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ HEIC conversion failed after {duration:.2f}s: {e}")
            raise
    return wrapper
```

## Best Practices

1. **Always provide fallback options** for HEIC files
2. **Show clear error messages** when HEIC processing fails
3. **Implement progressive enhancement** based on browser capabilities
4. **Monitor conversion performance** and optimize as needed
5. **Test with real HEIC files** from different devices
6. **Provide alternative upload methods** (drag-and-drop, camera)
7. **Cache converted images** to avoid repeated processing
8. **Validate file integrity** before and after conversion

## Deployment Considerations

- **Install HEIC libraries** in production environment
- **Configure appropriate timeouts** for conversion operations
- **Monitor memory usage** during HEIC processing
- **Implement rate limiting** to prevent abuse
- **Log conversion metrics** for monitoring and debugging
