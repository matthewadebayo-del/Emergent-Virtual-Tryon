import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
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
  CheckCircle
} from 'lucide-react';
import axios from 'axios';

const VirtualTryOn = ({ user, onLogout }) => {
  const [step, setStep] = useState(1);
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
  const [inputMode, setInputMode] = useState('catalog'); // 'catalog', 'upload', 'mixed'
  
  const fileInputRef = useRef(null);
  const clothingInputRef = useRef(null);

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await axios.get('/products');
      setProducts(response.data);
    } catch (error) {
      console.error('Failed to fetch products:', error);
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
    setSelectedProduct(product);
    setClothingImage(null);
    setClothingImagePreview(null);
  };

  const startTryOn = async () => {
    if (!userImage) {
      alert('Please upload your photo first');
      return;
    }

    if (!selectedProduct && !clothingImage) {
      alert('Please select a product or upload clothing image');
      return;
    }

    setLoading(true);
    setLoadingStep('Analyzing your photo...');

    try {
      // Convert image to base64 without data URL prefix
      const userImageBase64 = userImage.split(',')[1];
      const clothingImageBase64 = clothingImage ? clothingImage.split(',')[1] : null;

      setLoadingStep('Generating virtual try-on...');

      const requestData = {
        user_image_base64: userImageBase64,
        product_id: selectedProduct?.id || null,
        clothing_image_base64: clothingImageBase64,
        use_stored_measurements: useStoredMeasurements && user.measurements
      };

      const response = await axios.post('/tryon', requestData);
      
      setTryonResult(response.data);
      setStep(4);
    } catch (error) {
      console.error('Virtual try-on failed:', error);
      alert('Virtual try-on failed. Please try again.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const resetTryOn = () => {
    setStep(1);
    setUserImage(null);
    setUserImagePreview(null);
    setSelectedProduct(null);
    setClothingImage(null);
    setClothingImagePreview(null);
    setTryonResult(null);
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
            {[1, 2, 3, 4].map((stepNum) => (
              <div key={stepNum} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step >= stepNum 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-white/20 text-white/60'
                }`}>
                  {stepNum}
                </div>
                {stepNum < 4 && (
                  <div className={`w-16 h-1 mx-2 ${
                    step > stepNum ? 'bg-purple-600' : 'bg-white/20'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <div className="text-center text-white/80">
            {step === 1 && "Upload Your Photo"}
            {step === 2 && "Choose Input Mode"}
            {step === 3 && "Select Clothing"}
            {step === 4 && "Virtual Try-On Result"}
          </div>
        </div>

        {/* Step 1: User Photo Upload */}
        {step === 1 && (
          <div className="max-w-2xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">Upload Your Full Body Photo</h2>
              <p className="text-white/70 mb-8">
                Upload a clear, full-body photo taken from the front. This helps us create accurate measurements and better try-on results.
              </p>
              
              {!userImagePreview ? (
                <div 
                  className="upload-area"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Camera className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                  <p className="text-white/80 text-lg mb-2">Click to upload your photo</p>
                  <p className="text-white/60 text-sm">Supports JPG, PNG (Max 10MB)</p>
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

              {user.measurements && (
                <div className="mt-6 p-4 bg-green-500/20 rounded-lg">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-200 font-medium">Saved measurements found</span>
                  </div>
                  <label className="flex items-center justify-center space-x-3">
                    <input
                      type="checkbox"
                      checked={useStoredMeasurements}
                      onChange={(e) => setUseStoredMeasurements(e.target.checked)}
                      className="rounded border-green-400 text-green-600 focus:ring-green-500"
                    />
                    <span className="text-green-200">Use my saved measurements</span>
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
              
              <button
                onClick={() => setStep(1)}
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
                  <div className="product-grid">
                    {products.map((product) => (
                      <div
                        key={product.id}
                        onClick={() => handleProductSelect(product)}
                        className={`card-dark cursor-pointer transition-all ${
                          selectedProduct?.id === product.id 
                            ? 'ring-2 ring-purple-500 shadow-purple' 
                            : ''
                        }`}
                      >
                        <img
                          src={product.image_url}
                          alt={product.name}
                          className="w-full h-48 object-cover rounded-lg mb-4"
                        />
                        <h4 className="text-white font-semibold mb-2">{product.name}</h4>
                        <p className="text-white/70 text-sm mb-2">{product.description}</p>
                        <p className="text-purple-400 font-semibold">${product.price}</p>
                      </div>
                    ))}
                  </div>
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

              <div className="text-center">
                <button
                  onClick={startTryOn}
                  disabled={!selectedProduct && !clothingImage}
                  className="btn-primary text-lg px-8 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  Start Virtual Try-On
                </button>
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
                <p className="text-white/70">
                  Based on your measurements, we recommend this size for the best fit.
                </p>
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
      </div>
    </div>
  );
};

export default VirtualTryOn;