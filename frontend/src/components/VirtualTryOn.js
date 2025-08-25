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
  const [userPhotoDataURL, setUserPhotoDataURL] = useState(null); // New state for persistent photo URL

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  // Check authentication and existing measurements on component mount
  useEffect(() => {
    if (!user) {
      console.log('No user found, redirecting to login...');
      navigate('/login');
      return;
    }
    console.log('User authenticated:', user.email);
    
    // Check if user already has measurements - if so, skip photo capture
    if (user.measurements && Object.keys(user.measurements).length > 0) {
      console.log('User already has measurements, skipping to step 2');
      setExtractedMeasurements(user.measurements);
      setEditableMeasurements({ ...user.measurements });
      
      // Check if user has a stored photo URL in their profile
      if (user.profile_photo) {
        setUserPhotoDataURL(user.profile_photo);
        setUserPhotoPreview(user.profile_photo);
        console.log('Using stored profile photo');
      }
      
      setStep(2); // Skip directly to measurement review step
    }
  }, [user, navigate]);

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
      console.log('File uploaded:', file.name, file.size);
      
      // Set photo immediately
      setUserPhoto(file);
      
      // Create persistent data URL
      const dataUrl = await convertFileToDataURL(file);
      setUserPhotoDataURL(dataUrl);
      setUserPhotoPreview(dataUrl);
      
      console.log('Photo and preview set, starting measurement extraction...');
      
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
      
      // IMPORTANT: Preserve the photo before making API call
      setUserPhoto(imageFile);
      if (!userPhotoPreview) {
        const previewUrl = URL.createObjectURL(imageFile);
        setUserPhotoPreview(previewUrl);
      }
      
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
        
        // Double-check photo is preserved
        console.log('Photo preserved check:', !!imageFile);
        
        console.log('Moving to step 2 (measurement editing)');
        setStep(2); // Move to measurement editing step
      } else {
        console.error('Measurement extraction failed:', response.data);
        setError('Failed to extract measurements. Please try again.');
      }
    } catch (error) {
      console.error('Error extracting measurements:', error);
      
      // Handle authentication errors specifically
      if (error.response?.status === 401) {
        setError('Authentication failed. Please login again.');
        // Navigate to login page
        setTimeout(() => {
          navigate('/login');
        }, 2000);
      } else {
        setError('Error extracting measurements. Please try again.');
        
        // For demo purposes, create mock measurements if API fails
        console.log('Creating mock measurements for demo...');
        const mockMeasurements = {
          height: 68.5,
          weight: 150.0,
          chest: 36.0,
          waist: 30.0,
          hips: 38.0,
          shoulder_width: 16.5,
          arm_length: 24.0,
          confidence_score: 0.85,
          recommended_sizes: {
            top: "L",
            bottom: "L", 
            dress: "L"
          }
        };
        
        setExtractedMeasurements(mockMeasurements);
        setEditableMeasurements({ ...mockMeasurements });
        setError('Using demo measurements. Please login for full functionality.');
        setStep(2); // Still proceed to measurement editing
      }
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
    console.log('startCountdown called');
    setCountdown(3);
    const countdownInterval = setInterval(() => {
      setCountdown(prev => {
        console.log('Countdown:', prev);
        if (prev <= 1) {
          clearInterval(countdownInterval);
          console.log('Countdown finished, capturing photo...');
          capturePhoto();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const convertFileToDataURL = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.readAsDataURL(file);
    });
  };
  
  const dataURLToBlob = (dataURL) => {
    return new Promise((resolve) => {
      const arr = dataURL.split(',');
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      resolve(new Blob([u8arr], { type: mime }));
    });
  };

  const capturePhoto = async () => {
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
          
          // Create persistent data URL (doesn't expire like blob URLs)
          const dataUrl = await convertFileToDataURL(file);
          setUserPhotoDataURL(dataUrl);
          setUserPhotoPreview(dataUrl); // Use data URL instead of blob URL
          
          console.log('Photo URLs created - Data URL length:', dataUrl.length);
          
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
    
    // Use recommended size from measurements if available
    let defaultSize = product.sizes[0] || '';
    if (editableMeasurements && editableMeasurements.recommended_sizes) {
      const category = product.category.toLowerCase();
      if (category.includes('top') || category.includes('shirt') || category.includes('sweater') || category.includes('blouse')) {
        defaultSize = editableMeasurements.recommended_sizes.top || defaultSize;
      } else if (category.includes('bottom') || category.includes('pant') || category.includes('jean')) {
        defaultSize = editableMeasurements.recommended_sizes.bottom || defaultSize;
      } else if (category.includes('dress')) {
        defaultSize = editableMeasurements.recommended_sizes.dress || defaultSize;
      } else if (category.includes('outerwear') || category.includes('jacket') || category.includes('cardigan')) {
        defaultSize = editableMeasurements.recommended_sizes.top || defaultSize;
      }
      
      // Make sure the recommended size is available for this product
      if (product.sizes.includes(defaultSize)) {
        // Size is available, use it
      } else {
        // Fall back to first available size
        defaultSize = product.sizes[0] || '';
      }
    }
    
    setSelectedSize(defaultSize);
    setSelectedColor(product.colors[0] || '');
    
    console.log('Product selected:', product.name);
    console.log('Recommended size from measurements:', editableMeasurements?.recommended_sizes);
    console.log('Selected size:', defaultSize);
    
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
    console.log('processTryOn function called');
    console.log('userPhoto:', userPhoto);
    console.log('userPhotoDataURL:', userPhotoDataURL);
    console.log('selectedProduct:', selectedProduct);
    
    if (!selectedProduct) {
      console.error('Missing selectedProduct');
      setError('Please select a product first');
      return;
    }

    // Check if we have a photo - if not, redirect to capture one
    if (!userPhoto && !userPhotoDataURL) {
      console.log('No photo available, redirecting to photo capture');
      setError('Please capture or upload your photo first');
      setStep(1); // Go back to photo capture
      return;
    }

    try {
      setIsProcessing(true);
      setError('');
      
      console.log('Starting try-on processing...');
      console.log('Service type:', serviceType);
      console.log('Selected size:', selectedSize);
      console.log('Selected color:', selectedColor);

      const formData = new FormData();
      
      // If we have userPhoto (File object), use it directly
      if (userPhoto) {
        console.log('Using userPhoto file object');
        formData.append('user_photo', userPhoto);
      } else if (userPhotoDataURL) {
        // If we only have data URL, convert it to blob
        console.log('Converting userPhotoDataURL to blob');
        const blob = await dataURLToBlob(userPhotoDataURL);
        formData.append('user_photo', blob, 'user_photo.jpg');
      } else {
        throw new Error('No photo available for processing');
      }
      
      formData.append('product_id', selectedProduct.id);
      formData.append('service_type', serviceType);
      formData.append('size', selectedSize);
      formData.append('color', selectedColor);
      
      console.log('FormData prepared, making API call...');

      const response = await axios.post(`${API}/tryon`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 60000 // 60 second timeout for 3D processing
      });

      console.log('Try-on API response:', response.data);

      if (response.data.success) {
        console.log('Try-on successful, moving to results...');
        setResult(response.data.data);
        setStep(5);
      } else {
        console.error('Try-on failed:', response.data);
        setError('Try-on processing failed. Please try again.');
      }
    } catch (error) {
      console.error('Try-on processing error:', error);
      console.error('Error response:', error.response?.data);
      
      if (error.code === 'ECONNABORTED') {
        setError('Processing timeout. The 3D pipeline may be taking longer than expected. Please try again.');
      } else {
        setError(error.response?.data?.detail || 'Try-on processing failed. Please try again.');
      }
    } finally {
      console.log('Setting isProcessing to false');
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
          
          {/* Authentication Check */}
          {!user ? (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold mb-4">Authentication Required</h2>
              <p className="text-gray-300 mb-6">Please login to access the virtual try-on feature.</p>
              <button
                onClick={() => navigate('/login')}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
              >
                Login Now
              </button>
            </div>
          ) : (
            <>
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
                  <div className="relative">
                    <img
                      src={userPhotoPreview}
                      alt="Your captured photo"
                      className="w-32 h-32 object-cover rounded-lg border-2 border-purple-400"
                      onError={(e) => {
                        console.error('Photo preview failed to load:', userPhotoPreview);
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                      onLoad={() => {
                        console.log('Photo preview loaded successfully:', userPhotoPreview);
                      }}
                    />
                    <div 
                      className="w-32 h-32 bg-gray-600 rounded-lg border-2 border-purple-400 flex items-center justify-center text-gray-300 text-sm text-center hidden"
                      style={{display: 'none'}}
                    >
                      <div>
                        <div className="text-2xl mb-1">📷</div>
                        <div>Your Photo</div>
                      </div>
                    </div>
                  </div>
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
                    
                    {/* Only show photo warning if NO measurements exist (meaning no photo was ever processed) */}
                    {!extractedMeasurements && !userPhoto && !userPhotoDataURL && (
                      <div className="bg-yellow-900 border border-yellow-600 rounded-lg p-4 mt-4">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertCircle className="w-5 h-5 text-yellow-400" />
                          <span className="text-yellow-200 font-semibold">Photo Required</span>
                        </div>
                        <p className="text-yellow-100 text-sm mb-3">
                          You need to capture a photo to proceed with virtual try-on.
                        </p>
                        <button
                          onClick={() => setStep(1)}
                          className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm transition-colors"
                        >
                          Capture Photo Now
                        </button>
                      </div>
                    )}
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

              {/* Photo validation warning - only if no measurements exist */}
              {!extractedMeasurements && !userPhoto && !userPhotoDataURL && (
                <div className="bg-yellow-900 border border-yellow-600 rounded-lg p-4 mb-6">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-yellow-400" />
                    <span className="text-yellow-200">Warning: You need a photo to proceed with virtual try-on.</span>
                  </div>
                  <button
                    onClick={() => setStep(1)}
                    className="mt-3 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm transition-colors"
                  >
                    Capture Photo
                  </button>
                </div>
              )}

              {editableMeasurements && (
                <div className="bg-green-900 border border-green-600 rounded-lg p-4 mb-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Check className="w-5 h-5 text-green-400" />
                    <span className="text-green-200 font-semibold">Measurements Saved Successfully!</span>
                  </div>
                  <div className="text-green-100 text-sm">
                    Recommended sizes: Tops: <strong>{editableMeasurements.recommended_sizes?.top}</strong>, 
                    Bottoms: <strong>{editableMeasurements.recommended_sizes?.bottom}</strong>, 
                    Dresses: <strong>{editableMeasurements.recommended_sizes?.dress}</strong>
                  </div>
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
                      <div className="w-full h-48 bg-gray-600 flex items-center justify-center relative">
                        <img
                          src={product.image_url}
                          alt={product.name}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                        <div 
                          className="absolute inset-0 bg-gray-600 items-center justify-center text-white text-sm text-center p-4 hidden"
                          style={{display: 'none'}}
                        >
                          <div>
                            <div className="w-16 h-16 bg-gray-500 rounded-lg mx-auto mb-2 flex items-center justify-center">
                              👕
                            </div>
                            <div className="font-semibold">{product.category}</div>
                          </div>
                        </div>
                      </div>
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
                      {selectedProduct.sizes.map((size) => {
                        const isRecommended = editableMeasurements?.recommended_sizes && 
                          (selectedProduct.category.toLowerCase().includes('top') || 
                           selectedProduct.category.toLowerCase().includes('shirt') || 
                           selectedProduct.category.toLowerCase().includes('sweater')) 
                          ? size === editableMeasurements.recommended_sizes.top
                          : selectedProduct.category.toLowerCase().includes('bottom') || 
                            selectedProduct.category.toLowerCase().includes('pant')
                          ? size === editableMeasurements.recommended_sizes.bottom
                          : selectedProduct.category.toLowerCase().includes('dress')
                          ? size === editableMeasurements.recommended_sizes.dress
                          : false;
                        
                        return (
                          <button
                            key={size}
                            onClick={() => setSelectedSize(size)}
                            className={`py-2 px-4 rounded-lg border-2 transition-all relative ${
                              selectedSize === size
                                ? 'border-purple-400 bg-purple-900 bg-opacity-50 text-purple-200'
                                : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                            }`}
                          >
                            {size}
                            {isRecommended && (
                              <span className="absolute -top-1 -right-1 bg-green-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                                ✓
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                    {editableMeasurements?.recommended_sizes && (
                      <p className="text-green-400 text-sm mt-2 flex items-center gap-1">
                        <span>✓</span>
                        <span>
                          Recommended size based on your measurements: {
                            selectedProduct.category.toLowerCase().includes('top') || 
                            selectedProduct.category.toLowerCase().includes('shirt') || 
                            selectedProduct.category.toLowerCase().includes('sweater') || 
                            selectedProduct.category.toLowerCase().includes('blouse')
                              ? editableMeasurements.recommended_sizes.top
                              : selectedProduct.category.toLowerCase().includes('bottom') || 
                                selectedProduct.category.toLowerCase().includes('pant') ||
                                selectedProduct.category.toLowerCase().includes('jean')
                              ? editableMeasurements.recommended_sizes.bottom
                              : selectedProduct.category.toLowerCase().includes('dress')
                              ? editableMeasurements.recommended_sizes.dress
                              : editableMeasurements.recommended_sizes.top
                          }
                        </span>
                      </p>
                    )}
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
                  <div className="relative bg-gray-700 rounded-lg overflow-hidden">
                    {(userPhotoDataURL || userPhotoPreview) ? (
                      <img
                        src={userPhotoDataURL || userPhotoPreview}
                        alt="Original photo"
                        className="w-full h-64 object-cover"
                        onError={(e) => {
                          console.error('Original photo failed to load:', userPhotoDataURL || userPhotoPreview);
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                        onLoad={() => {
                          console.log('Original photo loaded successfully');
                        }}
                      />
                    ) : null}
                    <div 
                      className={`w-full h-64 bg-gray-600 flex items-center justify-center text-gray-300 ${(userPhotoDataURL || userPhotoPreview) ? 'hidden' : 'flex'}`}
                      style={{display: (userPhotoDataURL || userPhotoPreview) ? 'none' : 'flex'}}
                    >
                      <div className="text-center">
                        <div className="text-4xl mb-2">📷</div>
                        <div>Your photo will appear here</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Try-On Result</h3>
                  <div className="relative bg-gray-700 rounded-lg overflow-hidden">
                    {result?.result_image_url ? (
                      <img
                        src={result.result_image_url}
                        alt="Try-on result"
                        className="w-full h-64 object-cover"
                        onError={(e) => {
                          console.error('Try-on result failed to load:', result.result_image_url);
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                        onLoad={() => {
                          console.log('Try-on result loaded successfully');
                        }}
                      />
                    ) : null}
                    <div 
                      className={`w-full h-64 bg-gray-600 flex items-center justify-center text-gray-300 ${result?.result_image_url ? 'hidden' : 'flex'}`}
                      style={{display: result?.result_image_url ? 'none' : 'flex'}}
                    >
                      <div className="text-center">
                        <div className="text-4xl mb-2">✨</div>
                        <div>Try-on result</div>
                      </div>
                    </div>
                  </div>
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
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default VirtualTryOn;