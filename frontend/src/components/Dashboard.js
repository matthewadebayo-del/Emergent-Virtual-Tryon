import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { 
  Camera, 
  Upload, 
  User, 
  LogOut, 
  History, 
  Settings,
  Ruler,
  ShoppingBag,
  Zap,
  Crown
} from 'lucide-react';
import axios from 'axios';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('profile');
  const [showCamera, setShowCamera] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [measurements, setMeasurements] = useState(null);
  const [extractingMeasurements, setExtractingMeasurements] = useState(false);
  const [tryOnHistory, setTryOnHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  useEffect(() => {
    // Load user measurements if available
    if (user?.measurements) {
      setMeasurements(user.measurements);
    }

    // For new users, suggest camera capture first
    if (user && !user.measurements && !user.profile_photo) {
      setActiveTab('capture');
    }
  }, [user]);

  useEffect(() => {
    if (activeTab === 'history') {
      fetchTryOnHistory();
    }
  }, [activeTab]);

  const fetchTryOnHistory = async () => {
    try {
      setLoadingHistory(true);
      const response = await axios.get(`${API}/tryon-history`);
      if (response.data.success) {
        setTryOnHistory(response.data.data);
      }
    } catch (error) {
      console.error('Error fetching try-on history:', error);
    } finally {
      setLoadingHistory(false);
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
        
        // Add event listeners
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
          videoRef.current.play()
            .then(() => {
              console.log('Video playback started successfully');
            })
            .catch(error => {
              console.error('Error playing video:', error);
            });
        };
        
        videoRef.current.oncanplay = () => {
          console.log('Video can start playing');
        };
        
        videoRef.current.onerror = (error) => {
          console.error('Video element error:', error);
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
        alert('Video element not ready. Please try again.');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setShowCamera(false);
      alert(`Camera access failed: ${error.message}. Please check browser permissions and try again.`);
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
      
      canvas.toBlob(async (blob) => {
        if (blob) {
          setCapturing(true);
          await extractMeasurements(blob);
          setCapturing(false);
          stopCamera();
        }
      }, 'image/jpeg', 0.8);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setCapturing(true);
      await extractMeasurements(file);
      setCapturing(false);
    }
  };

  const extractMeasurements = async (imageFile) => {
    try {
      setExtractingMeasurements(true);
      
      const formData = new FormData();
      formData.append('user_photo', imageFile);
      
      const response = await axios.post(`${API}/extract-measurements`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.success) {
        setMeasurements(response.data.data);
        setActiveTab('measurements');
        
        // Show success message and suggest moving to try-on
        setTimeout(() => {
          if (window.confirm('Measurements extracted successfully! Would you like to start a virtual try-on now?')) {
            navigate('/tryon');
          }
        }, 2000);
      } else {
        alert('Failed to extract measurements. Please try again.');
      }
    } catch (error) {
      console.error('Error extracting measurements:', error);
      alert('Error extracting measurements. Please try again.');
    } finally {
      setExtractingMeasurements(false);
    }
  };

  const formatMeasurement = (value, unit = '"') => {
    return value ? `${value}${unit}` : 'N/A';
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const TabButton = ({ id, label, icon: Icon, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center gap-3 w-full px-4 py-3 rounded-lg transition-colors ${
        isActive 
          ? 'bg-purple-600 text-white' 
          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
      }`}
    >
      <Icon className="w-5 h-5" />
      {label}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-purple-400">Dashboard</h1>
            <div className="flex items-center gap-4">
              <span className="text-gray-300">Welcome, {user?.full_name}</span>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="space-y-2">
                <TabButton
                  id="profile"
                  label="Profile"
                  icon={User}
                  isActive={activeTab === 'profile'}
                  onClick={setActiveTab}
                />
                <TabButton
                  id="capture"
                  label="Capture Photo"
                  icon={Camera}
                  isActive={activeTab === 'capture'}
                  onClick={setActiveTab}
                />
                <TabButton
                  id="measurements"
                  label="Measurements"
                  icon={Ruler}
                  isActive={activeTab === 'measurements'}
                  onClick={setActiveTab}
                />
                <TabButton
                  id="tryon"
                  label="Virtual Try-On"
                  icon={ShoppingBag}
                  isActive={activeTab === 'tryon'}
                  onClick={setActiveTab}
                />
                <TabButton
                  id="history"
                  label="Try-On History"
                  icon={History}
                  isActive={activeTab === 'history'}
                  onClick={setActiveTab}
                />
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg p-6">
              {/* Profile Tab */}
              {activeTab === 'profile' && (
                <div>
                  <h2 className="text-2xl font-bold mb-6">Profile Information</h2>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-gray-300 mb-2">Full Name</label>
                      <input
                        type="text"
                        value={user?.full_name || ''}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white"
                        readOnly
                      />
                    </div>
                    <div>
                      <label className="block text-gray-300 mb-2">Email</label>
                      <input
                        type="email"
                        value={user?.email || ''}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white"
                        readOnly
                      />
                    </div>
                    <div>
                      <label className="block text-gray-300 mb-2">Member Since</label>
                      <input
                        type="text"
                        value={user?.created_at ? new Date(user.created_at).toLocaleDateString() : ''}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg py-3 px-4 text-white"
                        readOnly
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Capture Photo Tab */}
              {activeTab === 'capture' && (
                <div>
                  <h2 className="text-2xl font-bold mb-6">Capture Your Photo</h2>
                  <p className="text-gray-300 mb-6">
                    Take a full-body photo or upload an existing image to extract your measurements.
                  </p>

                  {!showCamera ? (
                    <div className="space-y-4">
                      <button
                        onClick={startCamera}
                        className="w-full bg-purple-600 hover:bg-purple-700 text-white py-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-3"
                      >
                        <Camera className="w-6 h-6" />
                        Start Camera (Recommended for first-time users)
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
                        Upload Photo
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-4">
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
                        {/* Debug info */}
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
                      
                      {/* Debug section */}
                      <div className="mt-4 text-sm text-gray-400 bg-gray-700 p-3 rounded">
                        <p><strong>Debug Info:</strong></p>
                        <p>Camera Status: {videoRef.current?.srcObject ? 'Connected' : 'Not Connected'}</p>
                        <p>Video Element: {videoRef.current ? 'Ready' : 'Not Ready'}</p>
                        <p>Show Camera: {showCamera ? 'True' : 'False'}</p>
                        <button
                          onClick={() => {
                            console.log('Video element:', videoRef.current);
                            console.log('Video srcObject:', videoRef.current?.srcObject);
                            console.log('Video readyState:', videoRef.current?.readyState);
                            console.log('Video videoWidth:', videoRef.current?.videoWidth);
                            console.log('Video videoHeight:', videoRef.current?.videoHeight);
                          }}
                          className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded"
                        >
                          Log Video Info to Console
                        </button>
                      </div>
                    </div>
                  )}

                  <canvas ref={canvasRef} className="hidden" />

                  {(capturing || extractingMeasurements) && (
                    <div className="mt-6 text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
                      <p className="text-gray-300">
                        {extractingMeasurements ? 'Extracting measurements...' : 'Processing image...'}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Measurements Tab */}
              {activeTab === 'measurements' && (
                <div>
                  <h2 className="text-2xl font-bold mb-6">Body Measurements</h2>
                  
                  {measurements ? (
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-purple-400">Basic Measurements</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-gray-300">Height:</span>
                            <span className="text-white font-semibold">
                              {formatMeasurement(measurements.height, '"')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Weight:</span>
                            <span className="text-white font-semibold">
                              {formatMeasurement(measurements.weight, ' lbs')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Chest:</span>
                            <span className="text-white font-semibold">
                              {formatMeasurement(measurements.chest, '"')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Waist:</span>
                            <span className="text-white font-semibold">
                              {formatMeasurement(measurements.waist, '"')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-300">Hips:</span>
                            <span className="text-white font-semibold">
                              {formatMeasurement(measurements.hips, '"')}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-purple-400">Size Recommendations</h3>
                        {measurements.recommended_sizes && (
                          <div className="space-y-3">
                            <div className="flex justify-between">
                              <span className="text-gray-300">Tops:</span>
                              <span className="text-white font-semibold">
                                {measurements.recommended_sizes.top}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Bottoms:</span>
                              <span className="text-white font-semibold">
                                {measurements.recommended_sizes.bottom}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Dresses:</span>
                              <span className="text-white font-semibold">
                                {measurements.recommended_sizes.dress}
                              </span>
                            </div>
                          </div>
                        )}
                        
                        <div className="mt-6">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-gray-300">Confidence Score:</span>
                            <span className="text-green-400 font-semibold">
                              {Math.round((measurements.confidence_score || 0) * 100)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-600 rounded-full h-2">
                            <div 
                              className="bg-green-500 h-2 rounded-full transition-all"
                              style={{ width: `${(measurements.confidence_score || 0) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <Ruler className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-400 mb-4">No measurements available</p>
                      <button
                        onClick={() => setActiveTab('capture')}
                        className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
                      >
                        Capture Photo to Get Measurements
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Virtual Try-On Tab */}
              {activeTab === 'tryon' && (
                <div>
                  <h2 className="text-2xl font-bold mb-6">Virtual Try-On</h2>
                  
                  <div className="grid md:grid-cols-2 gap-6 mb-8">
                    <div className="bg-gray-700 rounded-lg p-6 text-center">
                      <div className="bg-green-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                        <Zap className="w-8 h-8 text-green-400" />
                      </div>
                      <h3 className="text-xl font-bold text-green-400 mb-2">Hybrid 3D Approach</h3>
                      <p className="text-sm text-green-300 mb-4">Default • Cost-Effective</p>
                      <p className="text-gray-300 text-sm mb-4">
                        Advanced 3D body reconstruction with physics-based garment fitting
                      </p>
                      <div className="text-center mb-4">
                        <span className="text-2xl font-bold text-green-400">$0.02</span>
                        <span className="text-gray-400"> per try-on</span>
                      </div>
                      <button
                        onClick={() => navigate('/tryon?service=hybrid')}
                        className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-semibold transition-colors"
                      >
                        Start Hybrid Try-On
                      </button>
                    </div>

                    <div className="bg-gray-700 rounded-lg p-6 text-center border border-purple-400">
                      <div className="bg-purple-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                        <Crown className="w-8 h-8 text-purple-400" />
                      </div>
                      <h3 className="text-xl font-bold text-purple-400 mb-2">Premium fal.ai</h3>
                      <p className="text-sm text-purple-300 mb-4">Premium • Highest Quality</p>
                      <p className="text-gray-300 text-sm mb-4">
                        Professional-grade FASHN API with instant high-quality results
                      </p>
                      <div className="text-center mb-4">
                        <span className="text-2xl font-bold text-purple-400">$0.075</span>
                        <span className="text-gray-400"> per try-on</span>
                      </div>
                      <button
                        onClick={() => navigate('/tryon?service=premium')}
                        className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-semibold transition-colors"
                      >
                        Start Premium Try-On
                      </button>
                    </div>
                  </div>

                  {!measurements && (
                    <div className="bg-yellow-900 border border-yellow-600 rounded-lg p-4 mb-6">
                      <p className="text-yellow-200">
                        <strong>Tip:</strong> For the best try-on results, please capture your photo first to extract your measurements.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* History Tab */}
              {activeTab === 'history' && (
                <div>
                  <h2 className="text-2xl font-bold mb-6">Try-On History</h2>
                  
                  {loadingHistory ? (
                    <div className="text-center py-12">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
                      <p className="text-gray-300">Loading history...</p>
                    </div>
                  ) : tryOnHistory.length > 0 ? (
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {tryOnHistory.map((item) => (
                        <div key={item.id} className="bg-gray-700 rounded-lg overflow-hidden">
                          <img
                            src={item.result_image_url}
                            alt="Try-on result"
                            className="w-full h-48 object-cover"
                          />
                          <div className="p-4">
                            <div className="flex justify-between items-center mb-2">
                              <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                item.service_type === 'premium' 
                                  ? 'bg-purple-900 text-purple-200' 
                                  : 'bg-green-900 text-green-200'
                              }`}>
                                {item.service_type === 'premium' ? 'Premium' : 'Hybrid'}
                              </span>
                              <span className="text-gray-400 text-sm">
                                ${item.cost.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-white font-semibold mb-1">
                              {item.metadata?.product_name || 'Product'}
                            </p>
                            <p className="text-gray-400 text-sm">
                              {new Date(item.created_at).toLocaleDateString()}
                            </p>
                            <p className="text-gray-400 text-sm">
                              Processing: {item.processing_time.toFixed(1)}s
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <History className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-400 mb-4">No try-on history yet</p>
                      <button
                        onClick={() => setActiveTab('tryon')}
                        className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
                      >
                        Start Your First Try-On
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;