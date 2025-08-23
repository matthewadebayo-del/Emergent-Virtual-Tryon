import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { 
  ArrowLeft, 
  Camera, 
  Upload, 
  ShoppingBag, 
  Zap, 
  Crown,
  Check,
  AlertCircle,
  Loader,
  RefreshCw
} from 'lucide-react';
import axios from 'axios';

const VirtualTryOn = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initialServiceType = searchParams.get('service') || 'hybrid';

  const [step, setStep] = useState(1); // 1: Upload, 2: Edit Measurements, 3: Select Product, 4: Configure, 5: Results
  const [serviceType, setServiceType] = useState(initialServiceType);
  const [userPhoto, setUserPhoto] = useState(null);
  const [userPhotoPreview, setUserPhotoPreview] = useState(null);
  const [extractedMeasurements, setExtractedMeasurements] = useState(null);
  const [editableMeasurements, setEditableMeasurements] = useState(null);
  const [extractingMeasurements, setExtractingMeasurements] = useState(false);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [selectedSize, setSelectedSize] = useState('');
  const [selectedColor, setSelectedColor] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState(0);
  const [showCamera, setShowCamera] = useState(false);
  const [loadingProducts, setLoadingProducts] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      setLoadingProducts(true);
      const response = await axios.get(`${API}/products`);
      setProducts(response.data.products);
    } catch (error) {
      console.error('Error fetching products:', error);
      setError('Failed to load products');
    } finally {
      setLoadingProducts(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setUserPhoto(file);
      const previewUrl = URL.createObjectURL(file);
      setUserPhotoPreview(previewUrl);
      
      // Extract measurements after photo upload
      await extractMeasurements(file);
    }
  };

  const extractMeasurements = async (imageFile) => {
    console.log('extractMeasurements function called with file:', imageFile);
    try {
      setExtractingMeasurements(true);
      setError('');
      
      console.log('Setting extractingMeasurements to true');
      
      const formData = new FormData();
      formData.append('user_photo', imageFile);
      
      console.log('Making API call to extract measurements...');
      
      const response = await axios.post(`${API}/extract-measurements`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log('Measurement extraction response:', response.data);

      if (response.data.success) {
        const measurements = response.data.data;
        console.log('Extracted measurements:', measurements);
        
        setExtractedMeasurements(measurements);
        setEditableMeasurements({ ...measurements });
        
        console.log('Moving to step 2 (measurement editing)');
        setStep(2); // Move to measurement editing step
      } else {
        console.error('Measurement extraction failed:', response.data);
        setError('Failed to extract measurements. Please try again.');
      }
    } catch (error) {
      console.error('Error extracting measurements:', error);
      setError('Error extracting measurements. Please try again.');
    } finally {
      console.log('Setting extractingMeasurements to false');
      setExtractingMeasurements(false);
    }
  };

  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      
      // First set showCamera to true to ensure video element is rendered
      setShowCamera(true);
      
      // Wait for React to render the video element
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        } 
      });
      
      console.log('Camera stream obtained:', stream);
      
      if (videoRef.current) {
        console.log('Video element found, setting srcObject...');
        videoRef.current.srcObject = stream;
        
        // Force video attributes
        videoRef.current.autoplay = true;
        videoRef.current.playsInline = true;
        videoRef.current.muted = true;
        
        console.log('Video element setup complete');
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
          videoRef.current.play()
            .then(() => {
              console.log('Video playback started successfully');
            })
            .catch(error => {
              console.error('Error playing video:', error);
              setError('Failed to start video playback. Please try again or use file upload.');
            });
        };
        
        videoRef.current.onerror = (error) => {
          console.error('Video element error:', error);
          setError('Video error occurred. Please refresh and try again.');
        };
        
        // Force play attempt
        setTimeout(() => {
          if (videoRef.current) {
            console.log('Force play attempt...');
            videoRef.current.play().catch(console.error);
          }
        }, 500);
        
      } else {
        console.error('Video element not found!');
        setError('Video element not ready. Please try again.');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setShowCamera(false);
      setError(`Camera access failed: ${error.message}. Please check browser permissions and try again.`);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    setShowCamera(false);
    setCountdown(0);
  };

  const startCountdown = () => {
    setCountdown(3);
    const countdownInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownInterval);
          capturePhoto();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const capturePhoto = () => {
    console.log('capturePhoto function called');
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      console.log('Drawing image to canvas');
      
      canvas.toBlob(async (blob) => {
        if (blob) {
          console.log('Blob created, size:', blob.size);
          const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
          setUserPhoto(file);
          const previewUrl = URL.createObjectURL(blob);
          setUserPhotoPreview(previewUrl);
          stopCamera();
          
          console.log('Photo captured, starting measurement extraction...');
          // Extract measurements after photo capture
          await extractMeasurements(file);
        } else {
          console.error('Failed to create blob from canvas');
        }
      }, 'image/jpeg', 0.8);
    } else {
      console.error('Video or canvas element not found:', {
        video: !!videoRef.current,
        canvas: !!canvasRef.current
      });
    }
  };

  const selectProduct = (product) => {
    setSelectedProduct(product);
    setSelectedSize(product.sizes[0] || '');
    setSelectedColor(product.colors[0] || '');
    setStep(4); // Move to configuration step
  };

  const confirmMeasurements = () => {
    // Update recommended sizes based on edited measurements
    const updatedMeasurements = {
      ...editableMeasurements,
      recommended_sizes: calculateRecommendedSizes(editableMeasurements)
    };
    setEditableMeasurements(updatedMeasurements);
    setStep(3); // Move to product selection
  };

  const calculateRecommendedSizes = (measurements) => {
    // Simple size recommendation logic based on measurements
    const chest = measurements.chest || 0;
    const waist = measurements.waist || 0;
    
    let topSize = 'M';
    let bottomSize = 'M';
    
    if (chest < 34) topSize = 'S';
    else if (chest > 40) topSize = 'L';
    else if (chest > 44) topSize = 'XL';
    
    if (waist < 30) bottomSize = 'S';
    else if (waist > 34) bottomSize = 'L';
    else if (waist > 38) bottomSize = 'XL';
    
    return {
      top: topSize,
      bottom: bottomSize,
      dress: topSize
    };
  };

  const handleMeasurementChange = (field, value) => {
    const updatedMeasurements = {
      ...editableMeasurements,
      [field]: parseFloat(value) || 0
    };
    
    // Recalculate recommended sizes automatically when measurements change
    updatedMeasurements.recommended_sizes = calculateRecommendedSizes(updatedMeasurements);
    
    setEditableMeasurements(updatedMeasurements);
  };

  const processTryOn = async () => {
    if (!userPhoto || !selectedProduct) return;

    try {
      setIsProcessing(true);
      setError('');

      const formData = new FormData();
      formData.append('user_photo', userPhoto);
      formData.append('product_id', selectedProduct.id);
      formData.append('service_type', serviceType);
      formData.append('size', selectedSize);
      formData.append('color', selectedColor);

      const response = await axios.post(`${API}/tryon`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.success) {
        setResult(response.data.data);
        setStep(5);
      } else {
        setError('Try-on processing failed. Please try again.');
      }
    } catch (error) {
      console.error('Error processing try-on:', error);
      setError(error.response?.data?.detail || 'Try-on processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetTryOn = () => {
    setStep(1);
    setUserPhoto(null);
    setUserPhotoPreview(null);
    setExtractedMeasurements(null);
    setEditableMeasurements(null);
    setSelectedProduct(null);
    setSelectedSize('');
    setSelectedColor('');
    setResult(null);
    setError('');
    if (userPhotoPreview) {
      URL.revokeObjectURL(userPhotoPreview);
    }
  };

  const ServiceTypeToggle = () => (
    <div className="bg-gray-700 rounded-lg p-4 mb-6">
      <h3 className="text-lg font-semibold text-white mb-4">Choose Service Type</h3>
      <div className="grid md:grid-cols-2 gap-4">
        <button
          onClick={() => setServiceType('hybrid')}
          className={`p-4 rounded-lg border-2 transition-all ${
            serviceType === 'hybrid'
              ? 'border-green-400 bg-green-900 bg-opacity-50'
              : 'border-gray-600 bg-gray-800'
          }`}
        >
          <div className="flex items-center gap-3 mb-2">
            <Zap className="w-6 h-6 text-green-400" />
            <span className="text-lg font-semibold text-green-400">Hybrid 3D</span>
          </div>
          <p className="text-sm text-gray-300 mb-2">Advanced 3D reconstruction with physics-based fitting</p>
          <div className="text-center">
            <span className="text-xl font-bold text-green-400">$0.02</span>
            <span className="text-gray-400 ml-1">per try-on</span>
          </div>
        </button>

        <button
          onClick={() => setServiceType('premium')}
          className={`p-4 rounded-lg border-2 transition-all ${
            serviceType === 'premium'
              ? 'border-purple-400 bg-purple-900 bg-opacity-50'
              : 'border-gray-600 bg-gray-800'
          }`}
        >
          <div className="flex items-center gap-3 mb-2">
            <Crown className="w-6 h-6 text-purple-400" />
            <span className="text-lg font-semibold text-purple-400">Premium fal.ai</span>
          </div>
          <p className="text-sm text-gray-300 mb-2">Professional FASHN API with instant results</p>
          <div className="text-center">
            <span className="text-xl font-bold text-purple-400">$0.075</span>
            <span className="text-gray-400 ml-1">per try-on</span>
          </div>
        </button>
      </div>
    </div>
  );

  const StepIndicator = () => (
    <div className="flex items-center justify-center mb-8">
      {[1, 2, 3, 4, 5].map((stepNum) => (
        <React.Fragment key={stepNum}>
          <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
            step >= stepNum 
              ? 'bg-purple-600 text-white' 
              : 'bg-gray-600 text-gray-300'
          }`}>
            {step > stepNum ? <Check className="w-5 h-5" /> : stepNum}
          </div>
          {stepNum < 5 && (
            <div className={`w-12 h-1 mx-2 ${
              step > stepNum ? 'bg-purple-600' : 'bg-gray-600'
            }`} />
          )}
        </React.Fragment>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </button>
            <h1 className="text-2xl font-bold text-purple-400">Virtual Try-On</h1>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          <ServiceTypeToggle />
          <StepIndicator />

          {error && (
            <div className="bg-red-900 border border-red-600 text-red-200 p-4 rounded-lg mb-6 flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {/* Step 1: Upload Photo */}
          {step === 1 && (
            <div className="bg-gray-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">Upload Your Photo</h2>
              <p className="text-gray-300 text-center mb-8">
                Take a full-body photo or upload an existing image for the best try-on results.
              </p>

              {!showCamera ? (
                <div className="space-y-4 max-w-md mx-auto">
                  <button
                    onClick={startCamera}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white py-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-3"
                  >
                    <Camera className="w-6 h-6" />
                    Take Photo with Camera
                  </button>
                  
                  <div className="text-center text-gray-400">or</div>
                  
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full bg-gray-600 hover:bg-gray-700 text-white py-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-3"
                  >
                    <Upload className="w-6 h-6" />
                    Upload from Device
                  </button>
                </div>
              ) : (
                <div className="space-y-4 max-w-md mx-auto">
                  <div className="relative bg-gray-900 rounded-lg overflow-hidden border-2 border-gray-600">
                    <video
                      key={showCamera ? 'camera-active' : 'camera-inactive'}
                      ref={videoRef}
                      autoPlay={true}
                      playsInline={true}
                      muted={true}
                      controls={false}
                      className="w-full h-96 object-cover bg-gray-800"
                      style={{ minHeight: '384px' }}
                    />
                    {countdown > 0 && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                        <div className="text-6xl font-bold text-white animate-pulse">
                          {countdown}
                        </div>
                      </div>
                    )}
                    {showCamera && (
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                        Camera Active
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-4">
                    <button
                      onClick={startCountdown}
                      disabled={countdown > 0}
                      className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white py-3 rounded-lg font-semibold transition-colors"
                    >
                      {countdown > 0 ? `Capturing in ${countdown}...` : 'Capture Photo (3-2-1)'}
                    </button>
                    <button
                      onClick={stopCamera}
                      className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-semibold transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <canvas ref={canvasRef} className="hidden" />
            </div>
          )}

          {/* Step 2: Edit Measurements */}
          {step === 2 && (
            <div className="bg-gray-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">Review & Adjust Your Measurements</h2>
              
              {userPhotoPreview && (
                <div className="flex justify-center mb-6">
                  <img
                    src={userPhotoPreview}
                    alt="Your photo"
                    className="w-32 h-32 object-cover rounded-lg border-2 border-purple-400"
                  />
                </div>
              )}

              {extractingMeasurements ? (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
                  <p className="text-gray-300 text-lg">Extracting measurements using AI...</p>
                  <p className="text-gray-400 text-sm mt-2">This may take a few seconds</p>
                </div>
              ) : editableMeasurements ? (
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Measurements Form */}
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-purple-400 mb-4">Body Measurements</h3>
                    
                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Height (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.height || ''}
                        onChange={(e) => handleMeasurementChange('height', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 68.5"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Weight (pounds)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.weight || ''}
                        onChange={(e) => handleMeasurementChange('weight', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 150.0"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Chest (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.chest || ''}
                        onChange={(e) => handleMeasurementChange('chest', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 36.0"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Waist (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.waist || ''}
                        onChange={(e) => handleMeasurementChange('waist', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 30.0"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Hips (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.hips || ''}
                        onChange={(e) => handleMeasurementChange('hips', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 38.0"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Shoulder Width (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.shoulder_width || ''}
                        onChange={(e) => handleMeasurementChange('shoulder_width', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 16.5"
                      />
                    </div>

                    <div>
                      <label className="block text-gray-300 mb-2 font-semibold">Arm Length (inches)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={editableMeasurements.arm_length || ''}
                        onChange={(e) => handleMeasurementChange('arm_length', e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., 24.0"
                      />
                    </div>
                  </div>

                  {/* Recommended Sizes */}
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-purple-400 mb-4">Recommended Sizes</h3>
                    
                    <div className="bg-gray-700 rounded-lg p-6 space-y-4">
                      {editableMeasurements.recommended_sizes ? (
                        <>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-300 font-medium">Tops (Shirts, Sweaters):</span>
                            <span className="text-white font-bold text-lg bg-purple-600 px-3 py-1 rounded">
                              {editableMeasurements.recommended_sizes.top}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-300 font-medium">Bottoms (Pants, Jeans):</span>
                            <span className="text-white font-bold text-lg bg-purple-600 px-3 py-1 rounded">
                              {editableMeasurements.recommended_sizes.bottom}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-300 font-medium">Dresses:</span>
                            <span className="text-white font-bold text-lg bg-purple-600 px-3 py-1 rounded">
                              {editableMeasurements.recommended_sizes.dress}
                            </span>
                          </div>
                        </>
                      ) : (
                        <p className="text-gray-400">Sizes will be calculated when you adjust measurements</p>
                      )}
                      
                      <div className="mt-6 pt-4 border-t border-gray-600">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-gray-300">Confidence Score:</span>
                          <span className="text-green-400 font-semibold">
                            {Math.round((editableMeasurements.confidence_score || 0) * 100)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full transition-all"
                            style={{ width: `${(editableMeasurements.confidence_score || 0) * 100}%` }}
                          ></div>
                        </div>
                        <p className="text-gray-400 text-sm mt-2">
                          Adjust measurements above for more accurate size recommendations
                        </p>
                      </div>
                    </div>

                    <div className="bg-blue-900 border border-blue-600 rounded-lg p-4">
                      <h4 className="text-blue-200 font-semibold mb-2">💡 Measurement Tips</h4>
                      <ul className="text-blue-100 text-sm space-y-1">
                        <li>• Chest: Measure around the fullest part</li>
                        <li>• Waist: Measure at the narrowest point</li>
                        <li>• Hips: Measure around the widest part</li>
                        <li>• Take measurements over thin clothing</li>
                      </ul>
                    </div>

                    <button
                      onClick={confirmMeasurements}
                      className="w-full bg-purple-600 hover:bg-purple-700 text-white py-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-3"
                    >
                      <Check className="w-5 h-5" />
                      Save Measurements & Continue to Products
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="bg-red-900 border border-red-600 rounded-lg p-4 max-w-md mx-auto">
                    <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
                    <p className="text-red-200">Failed to extract measurements</p>
                    <button
                      onClick={() => setStep(1)}
                      className="mt-4 px-6 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Step 3: Select Product */}
          {step === 3 && (
            <div className="bg-gray-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">Select a Product</h2>
              
              {userPhotoPreview && (
                <div className="flex justify-center mb-6">
                  <img
                    src={userPhotoPreview}
                    alt="Your photo"
                    className="w-32 h-32 object-cover rounded-lg border-2 border-purple-400"
                  />
                </div>
              )}

              {loadingProducts ? (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
                  <p className="text-gray-300">Loading products...</p>
                </div>
              ) : (
                <div className="grid md:grid-cols-3 gap-6">
                  {products.map((product) => (
                    <div
                      key={product.id}
                      onClick={() => selectProduct(product)}
                      className="bg-gray-700 rounded-lg overflow-hidden cursor-pointer hover:ring-2 hover:ring-purple-400 transition-all"
                    >
                      <img
                        src={product.image_url}
                        alt={product.name}
                        className="w-full h-48 object-cover"
                      />
                      <div className="p-4">
                        <h3 className="font-semibold text-white mb-1">{product.name}</h3>
                        <p className="text-gray-400 text-sm mb-2">{product.brand}</p>
                        <p className="text-purple-400 font-bold">${product.price}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 4: Configure */}
          {step === 4 && selectedProduct && (
            <div className="bg-gray-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">Configure Your Try-On</h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <img
                    src={selectedProduct.image_url}
                    alt={selectedProduct.name}
                    className="w-full h-64 object-cover rounded-lg mb-4"
                  />
                  <h3 className="text-xl font-semibold text-white mb-2">{selectedProduct.name}</h3>
                  <p className="text-gray-400 mb-2">{selectedProduct.description}</p>
                  <p className="text-purple-400 font-bold text-lg">${selectedProduct.price}</p>
                </div>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-gray-300 mb-3 font-semibold">Size</label>
                    <div className="grid grid-cols-3 gap-2">
                      {selectedProduct.sizes.map((size) => (
                        <button
                          key={size}
                          onClick={() => setSelectedSize(size)}
                          className={`py-2 px-4 rounded-lg border-2 transition-all ${
                            selectedSize === size
                              ? 'border-purple-400 bg-purple-900 bg-opacity-50 text-purple-200'
                              : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                          }`}
                        >
                          {size}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-gray-300 mb-3 font-semibold">Color</label>
                    <div className="grid grid-cols-2 gap-2">
                      {selectedProduct.colors.map((color) => (
                        <button
                          key={color}
                          onClick={() => setSelectedColor(color)}
                          className={`py-2 px-4 rounded-lg border-2 transition-all ${
                            selectedColor === color
                              ? 'border-purple-400 bg-purple-900 bg-opacity-50 text-purple-200'
                              : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                          }`}
                        >
                          {color}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <h4 className="font-semibold text-white mb-2">Selected Configuration</h4>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Service:</span>
                        <span className={serviceType === 'premium' ? 'text-purple-400' : 'text-green-400'}>
                          {serviceType === 'premium' ? 'Premium fal.ai' : 'Hybrid 3D'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Size:</span>
                        <span className="text-white">{selectedSize}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Color:</span>
                        <span className="text-white">{selectedColor}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Cost:</span>
                        <span className="text-white font-semibold">
                          ${serviceType === 'premium' ? '0.075' : '0.020'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={processTryOn}
                    disabled={isProcessing}
                    className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white py-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-3"
                  >
                    {isProcessing ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        Processing... ({serviceType === 'premium' ? '~10s' : '~30s'})
                      </>
                    ) : (
                      <>
                        <ShoppingBag className="w-5 h-5" />
                        Start Virtual Try-On
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Results */}
          {step === 5 && result && (
            <div className="bg-gray-800 rounded-lg p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">Try-On Results</h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Your Photo</h3>
                  <img
                    src={userPhotoPreview}
                    alt="Original photo"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Try-On Result</h3>
                  <img
                    src={result.result_image_url}
                    alt="Try-on result"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                </div>
              </div>

              <div className="mt-8 bg-gray-700 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-white mb-4">Processing Details</h4>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Service Used:</span>
                      <span className={result.service_type === 'premium' ? 'text-purple-400' : 'text-green-400'}>
                        {result.service_type === 'premium' ? 'Premium fal.ai' : 'Hybrid 3D'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Processing Time:</span>
                      <span className="text-white">{result.processing_time.toFixed(1)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Cost:</span>
                      <span className="text-white">${result.cost.toFixed(3)}</span>
                    </div>
                  </div>
                  
                  {result.service_type === 'hybrid' && (
                    <div className="space-y-2">
                      <h5 className="text-sm font-semibold text-green-400">Hybrid 3D Pipeline Stages:</h5>
                      <div className="text-sm text-gray-300 space-y-1">
                        <div>✓ 3D Body Reconstruction</div>
                        <div>✓ Garment Physics Simulation</div>
                        <div>✓ Photorealistic Rendering</div>
                        <div>✓ AI Enhancement</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-6 flex gap-4">
                <button
                  onClick={resetTryOn}
                  className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-5 h-5" />
                  Try Another Item
                </button>
                <button
                  onClick={() => navigate('/dashboard')}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-semibold transition-colors"
                >
                  Back to Dashboard
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VirtualTryOn;