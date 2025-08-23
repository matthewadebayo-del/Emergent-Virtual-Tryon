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

  const [step, setStep] = useState(1); // 1: Upload, 2: Select Product, 3: Configure, 4: Results
  const [serviceType, setServiceType] = useState(initialServiceType);
  const [userPhoto, setUserPhoto] = useState(null);
  const [userPhotoPreview, setUserPhotoPreview] = useState(null);
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

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUserPhoto(file);
      const previewUrl = URL.createObjectURL(file);
      setUserPhotoPreview(previewUrl);
      setStep(2);
    }
  };

  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        } 
      });
      
      console.log('Camera stream obtained:', stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, starting playback...');
          videoRef.current.play()
            .then(() => {
              console.log('Video playback started successfully');
              setShowCamera(true);
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
        
        // Fallback: set showCamera after a delay
        setTimeout(() => {
          if (videoRef.current && videoRef.current.srcObject && !showCamera) {
            console.log('Fallback: setting showCamera to true');
            setShowCamera(true);
          }
        }, 1000);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
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
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
          setUserPhoto(file);
          const previewUrl = URL.createObjectURL(blob);
          setUserPhotoPreview(previewUrl);
          stopCamera();
          setStep(2);
        }
      }, 'image/jpeg', 0.8);
    }
  };

  const selectProduct = (product) => {
    setSelectedProduct(product);
    setSelectedSize(product.sizes[0] || '');
    setSelectedColor(product.colors[0] || '');
    setStep(3);
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
        setStep(4);
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
      {[1, 2, 3, 4].map((stepNum) => (
        <React.Fragment key={stepNum}>
          <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
            step >= stepNum 
              ? 'bg-purple-600 text-white' 
              : 'bg-gray-600 text-gray-300'
          }`}>
            {step > stepNum ? <Check className="w-5 h-5" /> : stepNum}
          </div>
          {stepNum < 4 && (
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
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
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

          {/* Step 2: Select Product */}
          {step === 2 && (
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

          {/* Step 3: Configure */}
          {step === 3 && selectedProduct && (
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

          {/* Step 4: Results */}
          {step === 4 && result && (
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