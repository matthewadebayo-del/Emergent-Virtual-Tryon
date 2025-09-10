import os
import subprocess
import tempfile
import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image, ExifTags
import logging
import mimetypes

# Configure logging
logger = logging.getLogger(__name__)

def convert_heic_to_jpeg(heic_base64_or_bytes) -> bytes:
    """
    Convert HEIC image (base64 string or bytes) to JPEG format with enhanced error handling
    
    Args:
        heic_base64_or_bytes: Base64 encoded HEIC image data or raw bytes
        
    Returns:
        JPEG image data as bytes
    """
    try:
        if isinstance(heic_base64_or_bytes, str):
            logger.info(f"Starting HEIC conversion from base64, input length: {len(heic_base64_or_bytes)}")
            
            # Validate base64 input
            if not heic_base64_or_bytes:
                raise ValueError("Empty HEIC base64 data provided")
            
            missing_padding = len(heic_base64_or_bytes) % 4
            if missing_padding:
                heic_base64_or_bytes += '=' * (4 - missing_padding)
                logger.info(f"Added {4 - missing_padding} padding characters")
            
            try:
                heic_bytes = base64.b64decode(heic_base64_or_bytes)
                logger.info(f"Base64 decoded successfully: {len(heic_bytes)} bytes")
            except Exception as decode_error:
                raise ValueError(f"Base64 decoding failed: {str(decode_error)}")
        else:
            heic_bytes = heic_base64_or_bytes
            logger.info(f"Starting HEIC conversion from bytes: {len(heic_bytes)} bytes")
        
        # Validate minimum file size
        if len(heic_bytes) < 100:
            raise ValueError(f"HEIC data too small: {len(heic_bytes)} bytes")
        
        try:
            import pillow_heif
            heif_file = pillow_heif.open_heif(heic_bytes)
            logger.info(f"HEIC file opened successfully: {heif_file.size} {heif_file.mode}")
            
            # Convert to PIL Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            logger.info(f"PIL Image created: {image.size} {image.mode}")
            
        except Exception as heic_error:
            logger.warning(f"HEIC processing failed, trying as regular image: {str(heic_error)}")
            try:
                image = Image.open(io.BytesIO(heic_bytes))
                logger.info(f"Fallback image opened: {image.format} {image.size} {image.mode}")
            except Exception as img_error:
                raise ValueError(f"Cannot open as HEIC or regular image: HEIC error: {str(heic_error)}, Image error: {str(img_error)}")
        
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA', 'P'):
            logger.info(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Convert to JPEG
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=85, optimize=True)
        jpeg_bytes = output_buffer.getvalue()
        logger.info(f"JPEG conversion complete: {len(jpeg_bytes)} bytes")
        
        return jpeg_bytes
        
    except Exception as e:
        logger.error(f"HEIC to JPEG conversion failed: {str(e)}")
        raise Exception(f"HEIC to JPEG conversion failed: {str(e)}")

class HEICProcessor:
    """
    Robust HEIC file processor with multiple conversion methods
    Handles HEIC/HEIF files from iOS devices with proper orientation
    """
    
    def __init__(self):
        self.conversion_methods = []
        self._detect_available_methods()
    
    def _detect_available_methods(self):
        """Detect available HEIC conversion methods"""
        
        # Method 1: pillow-heif (recommended)
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            self.conversion_methods.append("pillow_heif")
            logger.info("✅ pillow-heif available")
        except ImportError:
            logger.warning("⚠️ pillow-heif not available")
        
        # Method 2: pyheif (backup)
        try:
            import pyheif
            self.conversion_methods.append("pyheif")
            logger.info("✅ pyheif available")
        except ImportError:
            logger.warning("⚠️ pyheif not available")
        
        # Method 3: ImageIO with pillow-heif backend
        try:
            import imageio
            self.conversion_methods.append("imageio")
            logger.info("✅ imageio available")
        except ImportError:
            logger.warning("⚠️ imageio not available")
        
        # Method 4: External tool (heif-convert)
        if self._check_heif_convert():
            self.conversion_methods.append("heif_convert")
            logger.info("✅ heif-convert available")
        else:
            logger.warning("⚠️ heif-convert not available")
        
        # Method 5: FFmpeg (last resort)
        if self._check_ffmpeg():
            self.conversion_methods.append("ffmpeg")
            logger.info("✅ ffmpeg available")
        else:
            logger.warning("⚠️ ffmpeg not available")
        
        if not self.conversion_methods:
            logger.error("❌ No HEIC conversion methods available!")
        else:
            logger.info(f"Available conversion methods: {self.conversion_methods}")
    
    def _check_heif_convert(self) -> bool:
        """Check if heif-convert is available"""
        try:
            result = subprocess.run(['heif-convert', '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg with HEIF support is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def is_heic_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is HEIC/HEIF format"""
        file_path = Path(file_path)
        
        # Check by extension
        ext = file_path.suffix.lower()
        if ext in ['.heic', '.heif']:
            return True
        
        # Check by MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type in ['image/heic', 'image/heif']:
            return True
        
        # Check by reading file header
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
                # HEIF files start with specific byte patterns
                if b'ftyp' in header and (b'heic' in header or b'mif1' in header):
                    return True
        except Exception:
            pass
        
        return False
    
    def convert_heic_pillow_heif(self, input_path: str, output_path: str) -> bool:
        """Convert HEIC using pillow-heif (recommended method)"""
        try:
            import pillow_heif
            
            # Read HEIC file
            heif_file = pillow_heif.read_heif(input_path)
            
            # Convert to PIL Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            
            # Copy metadata
            if hasattr(heif_file, 'info'):
                image.info.update(heif_file.info)
            
            # Handle orientation
            image = self._fix_orientation(image)
            
            # Save as JPEG
            image.save(output_path, "JPEG", quality=95, optimize=True)
            logger.info(f"✅ Converted HEIC using pillow-heif: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ pillow-heif conversion failed: {e}")
            return False
    
    def _fix_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data"""
        try:
            exif = image._getexif()
            if exif is not None:
                orientation_key = None
                for tag, value in ExifTags.TAGS.items():
                    if value == 'Orientation':
                        orientation_key = tag
                        break
                
                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]
                    
                    # Apply rotation based on orientation
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 4:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 7:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    
                    logger.debug(f"Applied orientation correction: {orientation}")
        
        except Exception as e:
            logger.warning(f"⚠️ Could not fix orientation: {e}")
        
        return image
    
    def convert_heic(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert HEIC file to JPEG using the best available method
        
        Args:
            input_path: Path to HEIC file
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to converted JPEG file or None if conversion failed
        """
        
        # Validate input
        if not os.path.exists(input_path):
            logger.error(f"❌ Input file not found: {input_path}")
            return None
        
        if not self.is_heic_file(input_path):
            logger.info(f"ℹ️ File is not HEIC, returning original: {input_path}")
            return input_path
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.with_suffix('.jpg'))
        
        # Try conversion methods in order of preference
        conversion_functions = {
            "pillow_heif": self.convert_heic_pillow_heif,
        }
        
        for method in self.conversion_methods:
            if method in conversion_functions:
                logger.info(f"Trying conversion method: {method}")
                
                if conversion_functions[method](input_path, output_path):
                    # Validate output
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"✅ HEIC conversion successful: {output_path}")
                        return output_path
                    else:
                        logger.warning(f"⚠️ {method} created invalid output")
        
        logger.error(f"❌ All HEIC conversion methods failed for: {input_path}")
        return None
    
    def process_uploaded_file(self, file_path: str, preserve_original: bool = False) -> str:
        """
        Process uploaded file, converting HEIC if necessary
        
        Args:
            file_path: Path to uploaded file
            preserve_original: Keep original file after conversion
            
        Returns:
            Path to processed (possibly converted) file
        """
        
        if not self.is_heic_file(file_path):
            logger.info(f"File is not HEIC, no conversion needed: {file_path}")
            return file_path
        
        # Generate converted file path
        converted_path = str(Path(file_path).with_suffix('.jpg'))
        
        # Convert HEIC to JPEG
        result_path = self.convert_heic(file_path, converted_path)
        
        if result_path:
            # Cleanup original file if requested
            if not preserve_original and result_path != file_path:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed original HEIC file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove original file: {e}")
            
            return result_path
        else:
            logger.error(f"HEIC conversion failed, returning original: {file_path}")
            return file_path
