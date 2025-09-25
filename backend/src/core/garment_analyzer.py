"""
Garment Image Analysis Pipeline
Extracts visual features from garment images for realistic virtual try-on
"""

import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import cv2
from typing import Dict, List, Tuple, Any

class GarmentImageAnalyzer:
    """Analyzes garment images to extract colors, textures, patterns, and fabric properties"""
    
    def __init__(self):
        self.initialized = True
        
    def analyze_garment_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Main analysis function - extracts all visual features from garment image"""
        try:
            # Convert bytes to image array
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            # Run all analysis components
            colors = self._extract_dominant_colors(image_array)
            texture = self._analyze_texture(image_array)
            patterns = self._detect_patterns(image_array)
            silhouette = self._extract_silhouette(image_array)
            fabric_type = self._predict_fabric_type(texture)
            
            result = {
                "colors": colors,
                "texture": texture,
                "patterns": patterns,
                "silhouette": silhouette,
                "fabric_type": fabric_type,
                "analysis_success": True,
                "dominant_colors": colors.get("palette", [(128, 128, 128)]),
                "texture_features": texture
            }
            
            print(f"[GARMENT] Analysis complete - dominant_colors: {result['dominant_colors']}")
            return result
            
        except Exception as e:
            print(f"[GARMENT] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "colors": {"primary": (128, 128, 128), "palette": [(128, 128, 128)]},
                "texture": {"roughness": 0.5, "complexity": 0.5},
                "patterns": {"type": "solid", "detected": False},
                "silhouette": {"aspect_ratio": 1.0},
                "fabric_type": "cotton",
                "analysis_success": False,
                "dominant_colors": [(128, 128, 128)],
                "texture_features": {"roughness": 0.5, "complexity": 0.5}
            }
    
    def _extract_dominant_colors(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract dominant colors using K-means clustering"""
        try:
            print(f"[GARMENT] Image shape: {image_array.shape}")
            print(f"[GARMENT] Image min/max values: {image_array.min()}/{image_array.max()}")
            
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            print(f"[GARMENT] Total pixels: {len(pixels)}")
            
            # Check average color of entire image first
            avg_color = np.mean(pixels, axis=0)
            print(f"[GARMENT] Average color of entire image: {avg_color}")
            
            # Filter out very light/dark pixels (likely background)
            mask = np.all((pixels > 30) & (pixels < 225), axis=1)
            filtered_pixels = pixels[mask]
            print(f"[GARMENT] Filtered pixels (30-225 range): {len(filtered_pixels)}")
            
            if len(filtered_pixels) < 100:  # Fallback if too few pixels
                print(f"[GARMENT] Too few filtered pixels, using all pixels")
                filtered_pixels = pixels
            
            # Check if most pixels are white/light
            white_pixels = np.all(pixels > 200, axis=1)
            white_count = np.sum(white_pixels)
            print(f"[GARMENT] White pixels (>200): {white_count}/{len(pixels)} ({white_count/len(pixels)*100:.1f}%)")
            
            # Use K-means to find dominant colors
            n_colors = min(5, len(filtered_pixels) // 100)  # Adaptive number of colors
            if n_colors < 1:
                n_colors = 1
                
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            # Get colors and their frequencies
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            print(f"[GARMENT] K-means cluster centers: {colors}")
            
            # Calculate color frequencies
            unique_labels, counts = np.unique(labels, return_counts=True)
            color_freq = list(zip(colors, counts))
            color_freq.sort(key=lambda x: x[1], reverse=True)  # Sort by frequency
            
            print(f"[GARMENT] Color frequencies: {[(color, count) for color, count in color_freq]}")
            
            # Extract primary color and full palette
            primary_color = tuple(color_freq[0][0])
            palette = [tuple(color) for color, _ in color_freq]
            
            print(f"[GARMENT] Final primary color: {primary_color}")
            print(f"[GARMENT] Full palette: {palette}")
            
            return {
                "primary": primary_color,
                "palette": palette,
                "hex_primary": "#{:02x}{:02x}{:02x}".format(*primary_color),
                "confidence": len(filtered_pixels) / len(pixels)
            }
            
        except Exception as e:
            print(f"[GARMENT] Color extraction failed: {e}")
            return {
                "primary": (128, 128, 128),
                "palette": [(128, 128, 128)],
                "hex_primary": "#808080",
                "confidence": 0.0
            }
    
    def _analyze_texture(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze fabric surface characteristics"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate surface roughness (pixel variance)
            roughness = np.std(gray) / 255.0  # Normalize to 0-1
            
            # Calculate edge density using Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate texture complexity (gradient magnitude)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            complexity = np.mean(gradient_magnitude) / 255.0
            
            return {
                "roughness": float(roughness),
                "edge_density": float(edge_density),
                "complexity": float(complexity),
                "smooth_factor": 1.0 - roughness  # Inverse of roughness
            }
            
        except Exception as e:
            print(f"[GARMENT] Texture analysis failed: {e}")
            return {
                "roughness": 0.5,
                "edge_density": 0.5,
                "complexity": 0.5,
                "smooth_factor": 0.5
            }
    
    def _detect_patterns(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Detect stripes, checks, and prints"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Analyze horizontal patterns (vertical stripes)
            h_projection = np.mean(gray, axis=0)
            h_peaks = self._find_peaks(h_projection)
            
            # Analyze vertical patterns (horizontal stripes)
            v_projection = np.mean(gray, axis=1)
            v_peaks = self._find_peaks(v_projection)
            
            # Determine pattern type
            pattern_type = "solid"
            pattern_detected = False
            
            if len(h_peaks) > 3 and len(v_peaks) > 3:
                pattern_type = "checkered"
                pattern_detected = True
            elif len(h_peaks) > 3:
                pattern_type = "vertical_stripes"
                pattern_detected = True
            elif len(v_peaks) > 3:
                pattern_type = "horizontal_stripes"
                pattern_detected = True
            elif np.std(gray) > 50:  # High variance suggests print/pattern
                pattern_type = "print"
                pattern_detected = True
            
            return {
                "type": pattern_type,
                "detected": pattern_detected,
                "horizontal_peaks": len(h_peaks),
                "vertical_peaks": len(v_peaks),
                "complexity_score": float(np.std(gray) / 255.0)
            }
            
        except Exception as e:
            print(f"[GARMENT] Pattern detection failed: {e}")
            return {
                "type": "solid",
                "detected": False,
                "horizontal_peaks": 0,
                "vertical_peaks": 0,
                "complexity_score": 0.0
            }
    
    def _find_peaks(self, signal: np.ndarray, min_distance: int = 10) -> List[int]:
        """Find peaks in 1D signal for pattern detection"""
        try:
            # Simple peak detection
            peaks = []
            for i in range(1, len(signal) - 1):
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    # Check minimum distance from previous peaks
                    if not peaks or i - peaks[-1] >= min_distance:
                        peaks.append(i)
            return peaks
        except:
            return []
    
    def _extract_silhouette(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract garment shape properties"""
        try:
            # Convert to grayscale and threshold to get garment silhouette
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's thresholding to separate garment from background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (main garment)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 1.0
                
                # Calculate area ratio
                contour_area = cv2.contourArea(largest_contour)
                total_area = image_array.shape[0] * image_array.shape[1]
                area_ratio = contour_area / total_area
                
                return {
                    "aspect_ratio": float(aspect_ratio),
                    "area_ratio": float(area_ratio),
                    "width": int(w),
                    "height": int(h),
                    "garment_detected": True
                }
            else:
                return {
                    "aspect_ratio": 1.0,
                    "area_ratio": 0.5,
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "garment_detected": False
                }
                
        except Exception as e:
            print(f"[GARMENT] Silhouette extraction failed: {e}")
            return {
                "aspect_ratio": 1.0,
                "area_ratio": 0.5,
                "width": 512,
                "height": 512,
                "garment_detected": False
            }
    
    def _predict_fabric_type(self, texture_data: Dict[str, float]) -> str:
        """Classify fabric type from texture characteristics"""
        try:
            roughness = texture_data.get("roughness", 0.5)
            edge_density = texture_data.get("edge_density", 0.5)
            complexity = texture_data.get("complexity", 0.5)
            
            # Classification rules based on texture characteristics
            if roughness < 0.3 and edge_density < 0.2:
                return "silk"  # Very smooth, low edges
            elif roughness > 0.7 and complexity > 0.6:
                return "wool"  # High roughness, complex texture
            elif edge_density > 0.5 and roughness > 0.4:
                return "denim"  # High edge density, medium roughness
            elif roughness < 0.4 and complexity < 0.4:
                return "cotton"  # Medium smooth, low complexity
            else:
                return "synthetic"  # Default for mixed characteristics
                
        except Exception as e:
            print(f"[GARMENT] Fabric prediction failed: {e}")
            return "cotton"  # Safe default