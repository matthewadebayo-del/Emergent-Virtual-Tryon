import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  ArrowLeft, 
  Camera, 
  Upload, 
  Shirt, 
  Sparkles, 
  Download,
  RefreshCw,
  User,
  Image as ImageIcon,
  Zap,
  CheckCircle,
  Video,
  Square
} from 'lucide-react';
import axios from 'axios';

const VirtualTryOn = ({ user, onLogout }) => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const cameraFirst = queryParams.get('mode') === 'camera-first';
  
  const [step, setStep] = useState(cameraFirst ? 0 : 1); // Start with camera setup if camera-first
  const [userImage, setUserImage] = useState(null);
  const [userImagePreview, setUserImagePreview] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [clothingImage, setClothingImage] = useState(null);
  const [clothingImagePreview, setClothingImagePreview] = useState(null);
  const [products, setProducts] = useState([]);
  const [tryonResult, setTryonResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [useStoredMeasurements, setUseStoredMeasurements] = useState(true);
  const [inputMode, setInputMode] = useState('catalog');
  const [cameraStream, setCameraStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [measurements, setMeasurements] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [processingType, setProcessingType] = useState('default'); // 'default' or 'premium'
  
  const fileInputRef = useRef(null);
  const clothingInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    fetchProducts();
    if (cameraFirst) {
      startCamera();
    }
  }, [cameraFirst]);

  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [cameraStream]);

  const fetchProducts = async () => {
    try {
      console.log('Fetching products...');
      const response = await axios.get('/products');
      console.log('Products response:', response.data);
      setProducts(response.data);
    } catch (error) {
      console.error('Failed to fetch products:', error);
      // Set some default products if API fails
      setProducts([
        {
          id: 'fallback-1',
          name: 'Classic White T-Shirt',
          category: 'shirts',
          sizes: ['XS', 'S', 'M', 'L', 'XL'],
          image_url: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400',
          description: 'Comfortable cotton white t-shirt',
          price: 29.99
        }
      ]);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      setCameraStream(stream);
      setIsCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Failed to start camera:', error);
      alert('Failed to access camera. Please ensure you have given camera permissions.');
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
      setIsCameraActive(false);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    // Start countdown
    setIsCountingDown(true);
    setCountdown(3);

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(countdownInterval);
          // Take the photo after countdown
          setTimeout(() => {
            const canvas = canvasRef.current;
            const video = videoRef.current;
            const context = canvas.getContext('2d');

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);

            canvas.toBlob((blob) => {
              const reader = new FileReader();
              reader.onload = (e) => {
                const base64 = e.target.result;
                setUserImage(base64);
                setUserImagePreview(base64);
                stopCamera();
                
                // Auto-extract measurements (simulated)
                extractMeasurementsFromImage();
                
                setStep(2); // Skip to input mode selection
                setIsCountingDown(false);
                setCountdown(null);
              };
              reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.8);
          }, 500);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const extractMeasurementsFromImage = () => {
    // Simulated measurement extraction - in production, this would use AI
    const simulatedMeasurements = {
      height: 170 + Math.random() * 20,
      weight: 65 + Math.random() * 15,
      chest: 85 + Math.random() * 15,
      waist: 70 + Math.random() * 15,
      hips: 90 + Math.random() * 15,
      shoulder_width: 40 + Math.random() * 10
    };
    
    setMeasurements(simulatedMeasurements);
    
    // Save measurements to backend
    saveMeasurementsToBackend(simulatedMeasurements);
  };

  const saveMeasurementsToBackend = async (measurementData) => {
    try {
      await axios.post('/measurements', measurementData);
      console.log('Measurements saved automatically');
    } catch (error) {
      console.error('Failed to save measurements:', error);
    }
  };

  const handleImageUpload = (file, type) => {
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target.result;
        if (type === 'user') {
          setUserImage(base64);
          setUserImagePreview(base64);
        } else if (type === 'clothing') {
          setClothingImage(base64);
          setClothingImagePreview(base64);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleProductSelect = (product) => {
    console.log('Selected product:', product);
    setSelectedProduct(product);
    setClothingImage(null);
    setClothingImagePreview(null);
    
    // Visual feedback for selection
    const productElement = document.querySelector(`[data-product-id="${product.id}"]`);
    if (productElement) {
      productElement.classList.add('product-selected');
    }
  };

  const startTryOn = async () => {
    console.log('Starting try-on with:', { selectedProduct, userImage: !!userImage, clothingImage: !!clothingImage });
    
    if (!userImage) {
      alert('Please capture or upload your photo first');
      return;
    }

    if (!selectedProduct && !clothingImage) {
      alert('Please select a product or upload clothing image');
      return;
    }

    setLoading(true);
    setLoadingStep('Analyzing your photo and creating avatar...');

    try {
      const userImageBase64 = userImage.split(',')[1];
      const clothingImageBase64 = clothingImage ? clothingImage.split(',')[1] : null;

      setLoadingStep('Generating virtual try-on with personalized avatar...');

      // Create FormData for the request
      const formData = new FormData();
      formData.append('user_image_base64', userImageBase64);
      formData.append('product_id', selectedProduct?.id || '');
      formData.append('clothing_image_base64', clothingImageBase64 || '');
      formData.append('use_stored_measurements', String(useStoredMeasurements && (user.measurements || measurements)));
      formData.append('processing_type', processingType);

      console.log('Sending try-on FormData with product_id:', selectedProduct?.id);
      console.log('Processing type:', processingType);

      const response = await axios.post('/tryon', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Try-on response:', response.data);
      setTryonResult(response.data);
      setStep(4);
    } catch (error) {
      console.error('Virtual try-on failed:', error);
      console.error('Error details:', error.response?.data);
      
      let errorMessage = 'Virtual try-on failed. Please try again.';
      if (error.response?.status === 422) {
        errorMessage = 'Invalid data provided. Please check your inputs and try again.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      alert(errorMessage);
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const resetTryOn = () => {
    setStep(cameraFirst ? 0 : 1);
    setUserImage(null);
    setUserImagePreview(null);
    setSelectedProduct(null);
    setClothingImage(null);
    setClothingImagePreview(null);
    setTryonResult(null);
    setMeasurements(null);
    if (cameraFirst) {
      startCamera();
    }
  };

  const downloadResult = () => {
    if (tryonResult) {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${tryonResult.result_image_base64}`;
      link.download = 'virtual-tryon-result.png';
      link.click();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Navigation */}
      <nav className="glass-dark sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <Link to="/dashboard" className="text-white/60 hover:text-white transition-colors">
                <ArrowLeft className="w-6 h-6" />
              </Link>
              <div className="flex items-center space-x-2">
                <Shirt className="w-8 h-8 text-purple-400" />
                <span className="text-2xl font-bold gradient-text">VirtualFit</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-white/80">Virtual Try-On</span>
              <button
                onClick={onLogout}
                className="text-white/60 hover:text-white transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-4 mb-4">
            {(cameraFirst ? [0, 2, 3, 4] : [1, 2, 3, 4]).map((stepNum, index) => (
              <div key={stepNum} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step >= stepNum 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-white/20 text-white/60'
                }`}>
                  {cameraFirst ? (stepNum === 0 ? <Camera className="w-5 h-5" /> : index + 1) : stepNum}
                </div>
                {index < (cameraFirst ? 3 : 3) && (
                  <div className={`w-16 h-1 mx-2 ${
                    step > stepNum ? 'bg-purple-600' : 'bg-white/20'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="text-center text-white/80">
            {step === 0 && "Camera Setup & Capture"}
            {step === 1 && "Upload Your Photo"}
            {step === 2 && "Choose Input Mode"}
            {step === 3 && "Select Clothing"}
            {step === 4 && "Virtual Try-On Result"}
          </div>
        </div>

        {/* Step 0: Camera Capture (Camera-First Mode) */}
        {step === 0 && (
          <div className="max-w-4xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">Take Your Full Body Photo</h2>
              <p className="text-white/70 mb-8">
                Position yourself in good lighting, stand straight, and make sure your full body is visible. 
                We'll automatically extract your measurements and create your personalized avatar.
              </p>
              
              <div className="relative inline-block">
                {isCameraActive ? (
                  <div className="space-y-4">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full max-w-md mx-auto rounded-lg shadow-lg"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    <div className="flex space-x-4 justify-center">
                      <button
                        onClick={capturePhoto}
                        className="btn-primary flex items-center"
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        Capture Photo
                      </button>
                      <button
                        onClick={stopCamera}
                        className="btn-secondary flex items-center"
                      >
                        <Square className="w-5 h-5 mr-2" />
                        Stop Camera
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="upload-area max-w-md mx-auto">
                      <Video className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                      <p className="text-white/80 text-lg mb-4">Camera not active</p>
                      <button
                        onClick={startCamera}
                        className="btn-primary"
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        Start Camera
                      </button>
                    </div>
                    <div className="text-center">
                      <p className="text-white/60 mb-4">Or upload an existing photo</p>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="btn-secondary"
                      >
                        <Upload className="w-5 h-5 mr-2" />
                        Upload Photo
                      </button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          handleImageUpload(e.target.files[0], 'user');
                          if (e.target.files[0]) {
                            extractMeasurementsFromImage();
                            setStep(2);
                          }
                        }}
                        className="hidden"
                      />
                    </div>
                  </div>
                )}
              </div>

              {measurements && (
                <div className="mt-6 p-4 bg-green-500/20 rounded-lg">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-200 font-medium">Measurements Extracted</span>
                  </div>
                  <div className="text-green-200 text-sm">
                    Height: {measurements.height?.toFixed(0)}cm, 
                    Chest: {measurements.chest?.toFixed(0)}cm, 
                    Waist: {measurements.waist?.toFixed(0)}cm
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 1: User Photo Upload (Upload Mode) */}
        {step === 1 && (
          <div className="max-w-2xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">Upload Your Full Body Photo</h2>
              <p className="text-white/70 mb-8">
                Upload a clear, full-body photo taken from the front. This helps us create accurate measurements and better try-on results.
              </p>
              
              {!userImagePreview ? (
                <div>
                  <div 
                    className="upload-area"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Camera className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                    <p className="text-white/80 text-lg mb-2">Click to upload your photo</p>
                    <p className="text-white/60 text-sm">Supports JPG, PNG (Max 10MB)</p>
                  </div>
                  <div className="mt-6">
                    <button
                      onClick={startCamera}
                      className="btn-secondary"
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Use Camera Instead
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <img 
                    src={userImagePreview} 
                    alt="Your photo" 
                    className="max-h-96 mx-auto rounded-lg shadow-lg"
                  />
                  <div className="flex space-x-4 justify-center">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="btn-secondary"
                    >
                      Change Photo
                    </button>
                    <button
                      onClick={() => setStep(2)}
                      className="btn-primary"
                    >
                      Continue
                    </button>
                  </div>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => handleImageUpload(e.target.files[0], 'user')}
                className="hidden"
              />

              {(user.measurements || measurements) && (
                <div className="mt-6 p-4 bg-green-500/20 rounded-lg">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-200 font-medium">Measurements available</span>
                  </div>
                  <label className="flex items-center justify-center space-x-3">
                    <input
                      type="checkbox"
                      checked={useStoredMeasurements}
                      onChange={(e) => setUseStoredMeasurements(e.target.checked)}
                      className="rounded border-green-400 text-green-600 focus:ring-green-500"
                    />
                    <span className="text-green-200">Use my measurements for better accuracy</span>
                  </label>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Choose Input Mode */}
        {step === 2 && (
          <div className="max-w-4xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">How would you like to try on clothes?</h2>
              
              <div className="grid md:grid-cols-3 gap-6">
                <button
                  onClick={() => {
                    setInputMode('catalog');
                    setStep(3);
                  }}
                  className="card-dark hover-lift text-center p-6"
                >
                  <Shirt className="w-12 h-12 text-purple-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Product Catalog</h3>
                  <p className="text-white/70 text-sm">Choose from our curated collection of clothing items</p>
                </button>

                <button
                  onClick={() => {
                    setInputMode('upload');
                    setStep(3);
                  }}
                  className="card-dark hover-lift text-center p-6"
                >
                  <Upload className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Upload Clothing</h3>
                  <p className="text-white/70 text-sm">Upload your own clothing image to try on</p>
                </button>

                <button
                  onClick={() => {
                    setInputMode('mixed');
                    setStep(3);
                  }}
                  className="card-dark hover-lift text-center p-6"
                >
                  <Sparkles className="w-12 h-12 text-green-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Both Options</h3>
                  <p className="text-white/70 text-sm">Access both catalog and upload options</p>
                </button>
              </div>
              
              {/* Premium/Default Processing Selection */}
              <div className="mt-8 p-6 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-lg border border-purple-500/30">
                <h3 className="text-xl font-semibold text-white mb-4 text-center">Choose Processing Quality</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <label className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                    processingType === 'default' 
                      ? 'border-blue-500 bg-blue-500/20 shadow-lg shadow-blue-500/25' 
                      : 'border-white/20 bg-white/5 hover:border-blue-400'
                  }`}>
                    <input
                      type="radio"
                      name="processingType"
                      value="default"
                      checked={processingType === 'default'}
                      onChange={(e) => setProcessingType(e.target.value)}
                      className="sr-only"
                    />
                    <div className="text-center">
                      <div className="text-blue-400 mb-2">⚡</div>
                      <h4 className="text-lg font-semibold text-white mb-2">Default Processing</h4>
                      <p className="text-white/70 text-sm mb-2">Fast OpenAI DALL-E 3 generation</p>
                      <div className="text-green-400 font-medium">FREE</div>
                    </div>
                  </label>
                  
                  <label className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                    processingType === 'premium' 
                      ? 'border-purple-500 bg-purple-500/20 shadow-lg shadow-purple-500/25' 
                      : 'border-white/20 bg-white/5 hover:border-purple-400'
                  }`}>
                    <input
                      type="radio"
                      name="processingType"
                      value="premium"
                      checked={processingType === 'premium'}
                      onChange={(e) => setProcessingType(e.target.value)}
                      className="sr-only"
                    />
                    <div className="text-center">
                      <div className="text-purple-400 mb-2">🚀</div>
                      <h4 className="text-lg font-semibold text-white mb-2">Premium Processing</h4>
                      <p className="text-white/70 text-sm mb-2">Advanced fal.ai FASHN v1.6 with Identity Preservation</p>
                      <div className="text-purple-400 font-medium">$0.075 per generation</div>
                    </div>
                  </label>
                </div>
                
                {processingType === 'premium' && (
                  <div className="mt-4 p-3 bg-purple-600/20 rounded-lg border border-purple-500/50">
                    <div className="text-purple-200 text-sm">
                      <strong>Premium Features:</strong>
                      <ul className="mt-2 space-y-1 text-xs">
                        <li>• Multi-stage AI pipeline with pose detection</li>
                        <li>• Advanced identity preservation technology</li>
                        <li>• Segmentation-free processing</li>
                        <li>• Physics-aware fabric deformation</li>
                        <li>• Professional 864x1296 resolution output</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
              
              <button
                onClick={() => setStep(cameraFirst ? 0 : 1)}
                className="btn-secondary mt-6"
              >
                Back
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Select Clothing */}
        {step === 3 && (
          <div className="max-w-6xl mx-auto">
            <div className="card">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-white">Select Clothing to Try On</h2>
                <button
                  onClick={() => setStep(2)}
                  className="btn-secondary"
                >
                  Back
                </button>
              </div>

              {(inputMode === 'catalog' || inputMode === 'mixed') && (
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-white mb-4">Product Catalog</h3>
                  {products.length === 0 ? (
                    <div className="text-center py-8">
                      <p className="text-white/60">Loading products...</p>
                      <div className="spinner mx-auto mt-4"></div>
                    </div>
                  ) : (
                    <div className="product-grid">
                      {products.map((product) => (
                        <div
                          key={product.id}
                          data-product-id={product.id}
                          onClick={() => handleProductSelect(product)}
                          className={`card-dark cursor-pointer transition-all ${
                            selectedProduct?.id === product.id 
                              ? 'ring-4 ring-purple-500 ring-opacity-75 shadow-lg shadow-purple-500/50 transform scale-102' 
                              : 'hover:ring-2 hover:ring-purple-400'
                          }`}
                        >
                          <img
                            src={product.image_url}
                            alt={product.name}
                            className="w-full h-48 object-cover rounded-lg mb-4"
                            onError={(e) => {
                              e.target.src = 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400';
                            }}
                          />
                          <h4 className="text-white font-semibold mb-2">{product.name}</h4>
                          <p className="text-white/70 text-sm mb-2">{product.description}</p>
                          <p className="text-purple-400 font-semibold">${product.price}</p>
                          {selectedProduct?.id === product.id && (
                            <div className="mt-2 text-center">
                              <span className="inline-block bg-purple-500 text-white text-xs px-2 py-1 rounded-full">
                                Selected
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {(inputMode === 'upload' || inputMode === 'mixed') && (
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-white mb-4">Upload Your Own Clothing</h3>
                  {!clothingImagePreview ? (
                    <div 
                      className="upload-area"
                      onClick={() => clothingInputRef.current?.click()}
                    >
                      <ImageIcon className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                      <p className="text-white/80 text-lg mb-2">Click to upload clothing image</p>
                      <p className="text-white/60 text-sm">Supports JPG, PNG (Max 10MB)</p>
                    </div>
                  ) : (
                    <div className="flex justify-center">
                      <div className="relative">
                        <img 
                          src={clothingImagePreview} 
                          alt="Clothing" 
                          className="max-h-48 rounded-lg shadow-lg"
                        />
                        <button
                          onClick={() => {
                            setClothingImage(null);
                            setClothingImagePreview(null);
                          }}
                          className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  )}
                  <input
                    ref={clothingInputRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleImageUpload(e.target.files[0], 'clothing')}
                    className="hidden"
                  />
                </div>
              )}

              {/* Premium/Default Processing Selection */}
              <div className="mb-8 p-6 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-lg border border-purple-500/30">
                <h3 className="text-xl font-semibold text-white mb-4 text-center">Choose Processing Quality</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <label className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                    processingType === 'default' 
                      ? 'border-blue-500 bg-blue-500/20 shadow-lg shadow-blue-500/25' 
                      : 'border-white/20 bg-white/5 hover:border-blue-400'
                  }`}>
                    <input
                      type="radio"
                      name="processingType"
                      value="default"
                      checked={processingType === 'default'}
                      onChange={(e) => setProcessingType(e.target.value)}
                      className="sr-only"
                    />
                    <div className="text-center">
                      <div className="text-blue-400 mb-2">⚡</div>
                      <h4 className="text-lg font-semibold text-white mb-2">Default Processing</h4>
                      <p className="text-white/70 text-sm mb-2">Fast OpenAI DALL-E 3 generation</p>
                      <div className="text-green-400 font-medium">FREE</div>
                    </div>
                  </label>
                  
                  <label className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                    processingType === 'premium' 
                      ? 'border-purple-500 bg-purple-500/20 shadow-lg shadow-purple-500/25' 
                      : 'border-white/20 bg-white/5 hover:border-purple-400'
                  }`}>
                    <input
                      type="radio"
                      name="processingType"
                      value="premium"
                      checked={processingType === 'premium'}
                      onChange={(e) => setProcessingType(e.target.value)}
                      className="sr-only"
                    />
                    <div className="text-center">
                      <div className="text-purple-400 mb-2">🚀</div>
                      <h4 className="text-lg font-semibold text-white mb-2">Premium Processing</h4>
                      <p className="text-white/70 text-sm mb-2">Advanced fal.ai FASHN v1.6 with Identity Preservation</p>
                      <div className="text-purple-400 font-medium">$0.075 per generation</div>
                    </div>
                  </label>
                </div>
                
                {processingType === 'premium' && (
                  <div className="mt-4 p-3 bg-purple-600/20 rounded-lg border border-purple-500/50">
                    <div className="text-purple-200 text-sm">
                      <strong>Premium Features:</strong>
                      <ul className="mt-2 space-y-1 text-xs">
                        <li>• Multi-stage AI pipeline with pose detection</li>
                        <li>• Advanced identity preservation technology</li>
                        <li>• Segmentation-free processing</li>
                        <li>• Physics-aware fabric deformation</li>
                        <li>• Professional 864x1296 resolution output</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>

              <div className="text-center">
                <button
                  onClick={startTryOn}
                  disabled={(!selectedProduct && !clothingImage) || loading}
                  className="btn-primary text-lg px-8 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  {processingType === 'premium' ? 'Start Premium Try-On' : 'Start Virtual Try-On'}
                  {selectedProduct && (
                    <span className="ml-2 text-sm">
                      with {selectedProduct.name}
                    </span>
                  )}
                </button>
                {!selectedProduct && !clothingImage && (
                  <p className="text-white/60 text-sm mt-2">
                    Please select a product or upload clothing image to continue
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Results */}
        {step === 4 && tryonResult && (
          <div className="max-w-4xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">Your Virtual Try-On Result</h2>
              
              <div className="grid md:grid-cols-2 gap-8 mb-8">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Original</h3>
                  <img 
                    src={userImagePreview} 
                    alt="Original" 
                    className="w-full max-h-96 object-contain rounded-lg shadow-lg"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Virtual Try-On</h3>
                  <img 
                    src={`data:image/png;base64,${tryonResult.result_image_base64}`}
                    alt="Try-on result" 
                    className="w-full max-h-96 object-contain rounded-lg shadow-lg"
                  />
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-white mb-4">Size Recommendation</h3>
                <div className="text-3xl font-bold text-purple-400 mb-2">
                  Size {tryonResult.size_recommendation}
                </div>
                <p className="text-white/70 mb-4">
                  Based on your measurements, we recommend this size for the best fit.
                </p>
                {tryonResult.personalization_note && (
                  <div className="bg-blue-500/20 rounded-lg p-4 mt-4 border border-blue-500/50">
                    <div className="flex items-start space-x-3">
                      <div className="text-blue-400 mt-1">🚀</div>
                      <div>
                        <h4 className="text-blue-200 font-semibold mb-2">Advanced Virtual Try-On Technology</h4>
                        <p className="text-blue-200/90 text-sm mb-3">
                          ✨ {tryonResult.personalization_note}
                        </p>
                        
                        {tryonResult.technical_details && (
                          <div className="bg-blue-600/20 rounded p-3 mt-3">
                            <h5 className="text-blue-200 font-medium mb-2">Processing Pipeline:</h5>
                            <div className="grid grid-cols-2 gap-2 text-xs text-blue-200/80">
                              <div>✅ Photo Analysis & Segmentation</div>
                              <div>✅ Pose Estimation & Mapping</div>
                              <div>✅ Identity Preservation</div>
                              <div>✅ Garment Integration</div>
                              <div>✅ Realistic Blending</div>
                              <div>✅ Quality Enhancement</div>
                            </div>
                          </div>
                        )}
                        
                        <div className="mt-3 text-blue-200/70 text-xs">
                          <strong>Technology:</strong> Multi-stage AI pipeline with {tryonResult.technical_details?.pipeline_stages || 5} processing stages for maximum realism and identity preservation.
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex justify-center space-x-4">
                <button
                  onClick={downloadResult}
                  className="btn-primary"
                >
                  <Download className="w-5 h-5 mr-2" />
                  Download Result
                </button>
                <button
                  onClick={resetTryOn}
                  className="btn-secondary"
                >
                  <RefreshCw className="w-5 h-5 mr-2" />
                  Try Again
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
            <div className="card text-center">
              <div className="spinner mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-white mb-2">Processing...</h3>
              <p className="text-white/70">{loadingStep}</p>
            </div>
          </div>
        )}

        {/* Countdown Overlay */}
        {isCountingDown && countdown !== null && (
          <div className="countdown-display animate-countdown">
            {countdown > 0 ? countdown : 'SMILE!'}
          </div>
        )}
      </div>
    </div>
  );
};

export default VirtualTryOn;
