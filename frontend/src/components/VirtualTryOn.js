
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
  Square,
  HelpCircle
} from 'lucide-react';
import axios from 'axios';
import PhotoGuideModal from './PhotoGuideModal';

// 🔧 Frontend Fix for Virtual Try-On Display
const VirtualTryOnResult = ({ apiResponse }) => {
  const [debugMode, setDebugMode] = useState(false);
  const [imageError, setImageError] = useState(false);
  
  // Debug the API response
  console.log('🔍 API Response Debug:', {
    hasResult: !!apiResponse?.result,
    resultLength: apiResponse?.result?.length,
    resultType: typeof apiResponse?.result,
    firstChars: apiResponse?.result?.substring(0, 50)
  });

  // Handle image loading errors
  const handleImageError = (e) => {
    console.error('❌ Image failed to load:', e);
    setImageError(true);
  };

  const handleImageLoad = (e) => {
    console.log('✅ Image loaded successfully:', {
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
      src: e.target.src.substring(0, 100) + '...'
    });
    setImageError(false);
  };

  // Check if we have valid base64 data
  const hasValidResult = apiResponse?.result && 
                        typeof apiResponse.result === 'string' &&
                        apiResponse.result.length > 1000; // Should be substantial

  if (!hasValidResult) {
    return (
      <div className="bg-red-500/20 p-4 rounded-lg border border-red-500/50">
        <h3 className="text-red-200 font-semibold mb-2">❌ No valid result data</h3>
        <pre className="text-red-200/80 text-xs overflow-auto max-h-32">{JSON.stringify(apiResponse, null, 2)}</pre>
      </div>
    );
  }

  // Ensure proper base64 format
  const imageData = apiResponse.result.startsWith('data:') 
    ? apiResponse.result 
    : `data:image/jpeg;base64,${apiResponse.result}`;

  return (
    <div className="space-y-4">
      <div className="flex justify-center">
        <button 
          onClick={() => setDebugMode(!debugMode)}
          className="btn-secondary text-sm"
        >
          🔍 Toggle Debug Mode
        </button>
      </div>

      {debugMode && (
        <div className="bg-blue-500/20 p-4 rounded-lg border border-blue-500/50">
          <h4 className="text-blue-200 font-semibold mb-2">🔍 Debug Information</h4>
          <ul className="text-blue-200/80 text-sm space-y-1">
            <li>Result length: {apiResponse.result.length} characters</li>
            <li>Estimated size: {Math.round(apiResponse.result.length * 0.75 / 1024)}KB</li>
            <li>Starts with data URL: {apiResponse.result.startsWith('data:') ? '✅' : '❌'}</li>
            <li>Base64 format check: {/^[A-Za-z0-9+/]*={0,2}$/.test(apiResponse.result.replace('data:image/jpeg;base64,', '')) ? '✅' : '❌'}</li>
          </ul>
          
          <details className="mt-2">
            <summary className="text-blue-200 cursor-pointer">Raw Response (first 200 chars)</summary>
            <pre className="text-blue-200/70 text-xs mt-2 overflow-auto max-h-32">{JSON.stringify(apiResponse, null, 2).substring(0, 500)}...</pre>
          </details>
        </div>
      )}

      <div className="flex justify-center">
        {imageError ? (
          <div className="bg-red-500/20 p-6 rounded-lg border border-red-500/50 text-center">
            <h3 className="text-red-200 font-semibold mb-2">❌ Image Display Error</h3>
            <p className="text-red-200/80 text-sm mb-4">The API returned data but the image couldn't be displayed.</p>
            <button 
              onClick={() => {
                // Try to download the raw data
                const link = document.createElement('a');
                link.href = imageData;
                link.download = 'tryon-result.jpg';
                link.click();
              }}
              className="btn-primary text-sm"
            >
              💾 Download Raw Image
            </button>
          </div>
        ) : (
          <img
            src={imageData}
            alt="Virtual Try-On Result"
            className="max-w-full max-h-96 object-contain rounded-lg shadow-lg border-2 border-white/20"
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        )}
      </div>

      {/* Test direct base64 display */}
      {debugMode && (
        <div className="bg-purple-500/20 p-4 rounded-lg border border-purple-500/50">
          <h4 className="text-purple-200 font-semibold mb-2">🧪 Direct Base64 Test</h4>
          <img 
            src={`data:image/jpeg;base64,${apiResponse.result.replace(/^data:image\/[^;]+;base64,/, '')}`}
            alt="Direct Base64 Test"
            className="max-w-48 border border-red-500 rounded"
            onLoad={() => console.log('✅ Direct base64 test passed')}
            onError={() => console.log('❌ Direct base64 test failed')}
          />
        </div>
      )}
    </div>
  );
};

const VirtualTryOn = ({ user, onLogout }) => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const cameraFirst = queryParams.get('mode') === 'camera-first';
  const apparelSelection = queryParams.get('mode') === 'apparel-selection';
  
  const getInitialStep = () => {
    if (cameraFirst) return 0;
    if (apparelSelection && user.captured_image) return 2;
    if (user.measurements || user.captured_image) return 2;
    return 1;
  };
  
  const [step, setStep] = useState(getInitialStep());// Start with camera setup if camera-first
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
  const [userHeight, setUserHeight] = useState(''); // Height in cm for measurement reference
  const [showPhotoGuide, setShowPhotoGuide] = useState(false);
  
  const fileInputRef = useRef(null);
  const clothingInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    fetchProducts();
    
    // Check if user has measurements
    if (user.measurements) {
      setMeasurements(user.measurements);
    }

    if (apparelSelection && user.captured_image) {
      setUserImage(user.captured_image);
      setUserImagePreview(user.captured_image);
    }

    if (user.captured_image && !userImage) {
      let cleanImage = user.captured_image;
      // Clean base64 data if it contains Python byte notation
      if (cleanImage.includes("b'") || cleanImage.includes('b"')) {
        console.log('Cleaning Python byte notation from stored image');
        cleanImage = cleanImage.replace(/b['"]/, '').replace(/['"]$/, '');
        cleanImage = cleanImage.replace(/\\x[0-9a-fA-F]{2}/g, '');
        if (!cleanImage.startsWith('data:image/')) {
          cleanImage = 'data:image/jpeg;base64,' + cleanImage;
        }
      }
      setUserImage(cleanImage);
      setUserImagePreview(cleanImage);
    }

    if ((cameraFirst || (!user.captured_image && step === 0)) && step === 0) {
      startCamera();
    }
  }, [cameraFirst, apparelSelection, user.captured_image, user.measurements]);

  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [cameraStream]);

  useEffect(() => {
    if (isCameraActive && videoRef.current) {
      console.log('Camera active, checking video element...');
      console.log('Video srcObject:', videoRef.current.srcObject);
      console.log('Video readyState:', videoRef.current.readyState);
      console.log('Video paused:', videoRef.current.paused);
      console.log('Video dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
    }
  }, [isCameraActive]);

  useEffect(() => {
    if (isCameraActive && videoRef.current) {
      console.log('Camera active, checking video element...');
      console.log('Video srcObject:', videoRef.current.srcObject);
      console.log('Video readyState:', videoRef.current.readyState);
      console.log('Video paused:', videoRef.current.paused);
      console.log('Video dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
      
      if (videoRef.current.paused && videoRef.current.srcObject) {
        console.log('Video is paused, attempting to play...');
        videoRef.current.play().catch(e => console.log('Force play failed:', e));
      }
    }
  }, [isCameraActive]);

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
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          } 
        });
      } catch (error) {
        console.warn('Failed with ideal constraints, trying basic constraints:', error);
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' } 
          });
        } catch (fallbackError) {
          console.warn('Failed with user-facing camera, trying any camera:', fallbackError);
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: true 
          });
        }
      }
      
      setCameraStream(stream);
      setIsCameraActive(true);
      
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log('Video stream assigned:', stream);
          console.log('Video element:', videoRef.current);
          console.log('Stream tracks:', stream.getTracks());
          console.log('Stream active:', stream.active);
          
          videoRef.current.play()
            .then(() => {
              console.log('Video playing successfully');
              console.log('Video dimensions after play:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
            })
            .catch(e => {
              console.error('Video play failed:', e);
              if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.load();
                videoRef.current.play().catch(err => console.error('Video restart failed:', err));
              }
            });
        } else {
          console.error('Video ref not available when trying to assign stream');
        }
      }, 100);
      
    } catch (error) {
      console.error('Failed to start camera:', error);
      let errorMessage = 'Failed to access camera. ';
      if (error.name === 'NotFoundError') {
        errorMessage += 'No camera device found. Please ensure you have a camera connected.';
      } else if (error.name === 'NotAllowedError') {
        errorMessage += 'Camera access denied. Please allow camera permissions and try again.';
      } else if (error.name === 'NotSupportedError') {
        errorMessage += 'Camera not supported in this environment.';
      } else {
        errorMessage += 'Please ensure you have given camera permissions and try again.';
      }
      alert(errorMessage);
      setIsCameraActive(false);
      setCameraStream(null);
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
            
            if (!canvas || !video || video.videoWidth === 0 || video.videoHeight === 0) {
              console.error('Canvas or video not ready for capture');
              alert('Camera not ready. Please ensure camera is active and try again.');
              setIsCountingDown(false);
              setCountdown(null);
              return;
            }
            
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);

            canvas.toBlob((blob) => {
              if (!blob) {
                console.error('Failed to create blob from canvas');
                alert('Failed to capture photo. Please try again.');
                setIsCountingDown(false);
                setCountdown(null);
                return;
              }
              
              const reader = new FileReader();
              reader.onload = (e) => {
                const base64 = e.target.result;
                setUserImage(base64);
                setUserImagePreview(base64);
                stopCamera();
                
                // Auto-extract measurements (simulated)
                extractMeasurementsFromImage();
                
                saveCapturedImageToProfile(base64);
                
                setStep(1.5); // Go to measurement adjustment step
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

  const extractMeasurementsFromImage = async () => {
    try {
      const heightCm = userHeight ? parseFloat(userHeight) : 170; // Default to 170cm if not provided
      
      const formData = new FormData();
      formData.append('user_image_base64', userImage.split(',')[1]);
      formData.append('user_height_cm', heightCm.toString());
      
      const response = await axios.post('/extract_measurements', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data && response.data.measurements) {
        setMeasurements(response.data.measurements);
        saveMeasurementsToBackend(response.data.measurements);
      } else {
        const simulatedMeasurements = {
          height: heightCm / 2.54,
          weight: Math.round((140 + Math.random() * 40) * 100) / 100,
          gender: null,
          age_range: null,
          
          // Head/neck measurements
          head_circumference: Math.round((21 + Math.random() * 3) * 100) / 100,
          neck_circumference: Math.round((14 + Math.random() * 3) * 100) / 100,
          
          // Upper body measurements
          shoulder_width: Math.round((16 + Math.random() * 4) * 100) / 100,
          chest: Math.round((34 + Math.random() * 8) * 100) / 100,
          chest_circumference: Math.round((34 + Math.random() * 8) * 100) / 100,
          bust_circumference: Math.round((34 + Math.random() * 8) * 100) / 100,
          underbust_circumference: Math.round((30 + Math.random() * 6) * 100) / 100,
          waist: Math.round((28 + Math.random() * 8) * 100) / 100,
          waist_circumference: Math.round((28 + Math.random() * 8) * 100) / 100,
          arm_length: Math.round((22 + Math.random() * 4) * 100) / 100,
          forearm_length: Math.round((10 + Math.random() * 2) * 100) / 100,
          bicep_circumference: Math.round((11 + Math.random() * 3) * 100) / 100,
          wrist_circumference: Math.round((6 + Math.random() * 1) * 100) / 100,
          
          // Lower body measurements
          hips: Math.round((36 + Math.random() * 8) * 100) / 100,
          hip_circumference: Math.round((36 + Math.random() * 8) * 100) / 100,
          thigh_circumference: Math.round((20 + Math.random() * 4) * 100) / 100,
          knee_circumference: Math.round((14 + Math.random() * 2) * 100) / 100,
          calf_circumference: Math.round((13 + Math.random() * 2) * 100) / 100,
          ankle_circumference: Math.round((8 + Math.random() * 1) * 100) / 100,
          inseam_length: Math.round((30 + Math.random() * 4) * 100) / 100,
          outseam_length: Math.round((40 + Math.random() * 4) * 100) / 100,
          rise_length: Math.round((10 + Math.random() * 2) * 100) / 100,
          
          // Torso measurements
          torso_length: Math.round((24 + Math.random() * 4) * 100) / 100,
          back_length: Math.round((16 + Math.random() * 2) * 100) / 100,
          sleeve_length: Math.round((24 + Math.random() * 3) * 100) / 100
        };
        
        setMeasurements(simulatedMeasurements);
        saveMeasurementsToBackend(simulatedMeasurements);
      }
    } catch (error) {
      console.error('Failed to extract measurements:', error);
      
      // Fallback to simulated measurements
      const heightCm = userHeight ? parseFloat(userHeight) : 170;
      const simulatedMeasurements = {
        height: heightCm / 2.54,
        weight: Math.round((140 + Math.random() * 40) * 100) / 100,
        gender: null,
        age_range: null,
        
        // Head/neck measurements
        head_circumference: Math.round((21 + Math.random() * 3) * 100) / 100,
        neck_circumference: Math.round((14 + Math.random() * 3) * 100) / 100,
        
        // Upper body measurements
        shoulder_width: Math.round((16 + Math.random() * 4) * 100) / 100,
        chest: Math.round((34 + Math.random() * 8) * 100) / 100,
        chest_circumference: Math.round((34 + Math.random() * 8) * 100) / 100,
        bust_circumference: Math.round((34 + Math.random() * 8) * 100) / 100,
        underbust_circumference: Math.round((30 + Math.random() * 6) * 100) / 100,
        waist: Math.round((28 + Math.random() * 8) * 100) / 100,
        waist_circumference: Math.round((28 + Math.random() * 8) * 100) / 100,
        arm_length: Math.round((22 + Math.random() * 4) * 100) / 100,
        forearm_length: Math.round((10 + Math.random() * 2) * 100) / 100,
        bicep_circumference: Math.round((11 + Math.random() * 3) * 100) / 100,
        wrist_circumference: Math.round((6 + Math.random() * 1) * 100) / 100,
        
        // Lower body measurements
        hips: Math.round((36 + Math.random() * 8) * 100) / 100,
        hip_circumference: Math.round((36 + Math.random() * 8) * 100) / 100,
        thigh_circumference: Math.round((20 + Math.random() * 4) * 100) / 100,
        knee_circumference: Math.round((14 + Math.random() * 2) * 100) / 100,
        calf_circumference: Math.round((13 + Math.random() * 2) * 100) / 100,
        ankle_circumference: Math.round((8 + Math.random() * 1) * 100) / 100,
        inseam_length: Math.round((30 + Math.random() * 4) * 100) / 100,
        outseam_length: Math.round((40 + Math.random() * 4) * 100) / 100,
        rise_length: Math.round((10 + Math.random() * 2) * 100) / 100,
        
        // Torso measurements
        torso_length: Math.round((24 + Math.random() * 4) * 100) / 100,
        back_length: Math.round((16 + Math.random() * 2) * 100) / 100,
        sleeve_length: Math.round((24 + Math.random() * 3) * 100) / 100
      };
      
      setMeasurements(simulatedMeasurements);
      saveMeasurementsToBackend(simulatedMeasurements);
    }
  };

  const saveMeasurementsToBackend = async (measurementData) => {
    try {
      await axios.post('/measurements', measurementData);
      console.log('Measurements saved automatically');
    } catch (error) {
      console.error('Failed to save measurements:', error);
    }
  };

  const saveCapturedImageToProfile = async (imageBase64) => {
    try {
      await axios.post('/save_captured_image', {
        image_base64: imageBase64,
        measurements: measurements
      });
      console.log('Captured image saved to profile');
    } catch (error) {
      console.error('Failed to save captured image:', error);
    }
  };

  const handleImageUpload = (file, type) => {
    if (file) {
      console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);
      
      const fileName = file.name.toLowerCase();
      const isHeic = fileName.endsWith('.heic') || fileName.endsWith('.heif') || 
                     file.type === 'image/heic' || file.type === 'image/heif';
      
      if (!file.type.startsWith('image/') && !isHeic) {
        alert('Please select a valid image file (including HEIC format)');
        return;
      }
      
      if (file.size > 50 * 1024 * 1024) {
        alert('File size must be less than 50MB');
        return;
      }

      if (isHeic) {
        console.log('HEIC file detected, processing with enhanced validation...');
        handleHeicFile(file, type);
      } else {
        handleStandardImageFile(file, type);
      }
    } else {
      console.log('No file provided to handleImageUpload');
    }
  };

  const handleHeicFile = async (file, type) => {
    try {
      console.log('Processing HEIC file with client-side compression and backend conversion...');
      
      // Create a canvas to compress the image before sending to backend
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      const reader = new FileReader();
      reader.onload = async (e) => {
        const heicBase64 = e.target.result;
        console.log('HEIC file read, compressing before backend conversion...');
        
        try {
          img.onload = async () => {
            const maxWidth = 800;
            const maxHeight = 1200;
            let { width, height } = img;
            
            if (width > maxWidth || height > maxHeight) {
              const ratio = Math.min(maxWidth / width, maxHeight / height);
              width *= ratio;
              height *= ratio;
            }
            
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);
            
            const compressedBase64 = canvas.toDataURL('image/jpeg', 0.7);
            console.log(`HEIC compressed from ${heicBase64.length} to ${compressedBase64.length} bytes`);
            
            if (type === 'user') {
              setUserImage(compressedBase64);
              setUserImagePreview(compressedBase64);
              console.log('HEIC user image compressed and set successfully');
            } else if (type === 'clothing') {
              setClothingImage(compressedBase64);
              setClothingImagePreview(compressedBase64);
              console.log('HEIC clothing image compressed and set successfully');
            }
          };
          
          img.onerror = async () => {
            console.log('Client-side HEIC processing failed, trying backend conversion...');
            
            try {
              // Fallback to backend conversion
              const base64Data = heicBase64.split(',')[1] || heicBase64;
              
              const response = await axios.post('/v1/convert-heic', {
                heic_base64: base64Data
              });
              
              const result = response.data;
              let jpegBase64 = result.jpeg_base64;
              
              // Clean any Python byte notation that might be present
              if (jpegBase64.includes("b'") || jpegBase64.includes('b"')) {
                console.log('Cleaning Python byte notation from backend response');
                jpegBase64 = jpegBase64.replace(/b['"]/, '').replace(/['"]$/, '');
                if (!jpegBase64.startsWith('data:image/')) {
                  jpegBase64 = 'data:image/jpeg;base64,' + jpegBase64;
                }
              }
              
              console.log('HEIC converted via backend successfully');
              
              if (type === 'user') {
                setUserImage(jpegBase64);
                setUserImagePreview(jpegBase64);
                console.log('HEIC user image converted and set successfully');
              } else if (type === 'clothing') {
                setClothingImage(jpegBase64);
                setClothingImagePreview(jpegBase64);
                console.log('HEIC clothing image converted and set successfully');
              }
            } catch (backendError) {
              console.error('Backend HEIC conversion failed:', backendError);
              alert('HEIC file could not be processed. Please convert to JPG/PNG format or try a different image.');
            }
          };
          
          img.src = heicBase64;
          
        } catch (conversionError) {
          console.error('HEIC processing failed:', conversionError);
          alert('HEIC file could not be processed. Please convert to JPG/PNG format or try a different image.');
        }
      };
      
      reader.onerror = (error) => {
        console.error('HEIC file reading failed:', error);
        alert('HEIC file could not be processed. Please convert to JPG/PNG or try a different image.');
      };
      
      reader.readAsDataURL(file);
      
    } catch (error) {
      console.error('HEIC processing failed:', error);
      alert('HEIC file could not be processed. Please convert to JPG/PNG or try a different image.');
    }
  };

  const compressImage = (file, maxSizeKB = 800) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.onload = () => {
        // Calculate new dimensions to keep under size limit
        let { width, height } = img;
        const maxWidth = 1200;
        const maxHeight = 1600;
        
        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height);
          width *= ratio;
          height *= ratio;
        }
        
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        
        // Try different quality levels to get under size limit
        let quality = 0.8;
        let compressedBase64;
        
        do {
          compressedBase64 = canvas.toDataURL('image/jpeg', quality);
          const sizeKB = (compressedBase64.length * 0.75) / 1024; // Approximate size
          
          if (sizeKB <= maxSizeKB || quality <= 0.1) break;
          quality -= 0.1;
        } while (quality > 0.1);
        
        console.log(`Image compressed: ${file.size} bytes -> ${Math.round((compressedBase64.length * 0.75))} bytes (quality: ${quality})`);
        resolve(compressedBase64);
      };
      
      img.onerror = () => resolve(null);
      
      const reader = new FileReader();
      reader.onload = (e) => img.src = e.target.result;
      reader.readAsDataURL(file);
    });
  };

  const handleStandardImageFile = async (file, type) => {
    try {
      // Compress image if it's too large
      const fileSizeKB = file.size / 1024;
      let base64;
      
      if (fileSizeKB > 800) {
        console.log(`Large file detected (${Math.round(fileSizeKB)}KB), compressing...`);
        base64 = await compressImage(file, 800);
        if (!base64) {
          throw new Error('Compression failed');
        }
      } else {
        // File is small enough, read directly
        base64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
      }
      
      console.log('File processed successfully, base64 length:', base64.length);
      
      // Clean base64 data - remove Python byte string notation if present
      if (base64.includes("b'") || base64.includes('b"')) {
        console.log('Cleaning Python byte string notation from base64');
        base64 = base64.replace(/b['"]/, '').replace(/['"]$/, '');
        base64 = base64.replace(/\\x[0-9a-fA-F]{2}/g, '');
        if (!base64.startsWith('data:image/')) {
          base64 = 'data:image/jpeg;base64,' + base64;
        }
      }
      
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
    } catch (error) {
      console.error('File processing failed:', error);
      alert('Failed to process the selected file. Please try again.');
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
      
      if (userHeight && !isNaN(userHeight)) {
        formData.append('user_height_cm', parseFloat(userHeight).toString());
      }

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
              <div className="flex items-center justify-center mb-6">
                <h2 className="text-2xl font-bold text-white mr-4">Take Your Full Body Photo</h2>
                <button
                  onClick={() => setShowPhotoGuide(true)}
                  className="text-blue-400 hover:text-blue-300 transition-colors"
                  title="View photo guide"
                >
                  <HelpCircle className="w-6 h-6" />
                </button>
              </div>
              <p className="text-white/70 mb-4">
                Follow the positioning guide below to capture accurate full-body measurements. 
                Our AI will extract 20+ body measurements for precise virtual try-on fitting.
              </p>
              
              <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg max-w-3xl mx-auto">
                <h3 className="text-blue-200 font-medium mb-4">📏 Full Body Measurement Capture Guidelines</h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-100/80">
                  <div className="space-y-3">
                    <div>
                      <strong className="text-blue-200">📍 Body Position:</strong>
                      <ul className="mt-1 space-y-1 text-xs">
                        <li>• Stand straight, feet shoulder-width apart</li>
                        <li>• Arms slightly away from body (T-pose)</li>
                        <li>• Face camera directly, look straight ahead</li>
                        <li>• Wear form-fitting clothes (not loose)</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-blue-200">📐 Distance & Frame:</strong>
                      <ul className="mt-1 space-y-1 text-xs">
                        <li>• Stand 6-8 feet from camera</li>
                        <li>• Full body visible: head to feet</li>
                        <li>• Leave space around body edges</li>
                        <li>• Camera at chest height</li>
                      </ul>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <strong className="text-blue-200">💡 Lighting & Background:</strong>
                      <ul className="mt-1 space-y-1 text-xs">
                        <li>• Use natural daylight or bright indoor light</li>
                        <li>• Plain, contrasting background</li>
                        <li>• Avoid shadows on body</li>
                        <li>• No backlighting or glare</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-blue-200">🎯 For Best Measurements:</strong>
                      <ul className="mt-1 space-y-1 text-xs">
                        <li>• Shoulders, waist, hips clearly visible</li>
                        <li>• Arms and legs not overlapping body</li>
                        <li>• Stand on flat surface</li>
                        <li>• Remove bulky accessories</li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-blue-600/20 rounded border border-blue-500/30">
                  <p className="text-blue-200 text-xs text-center">
                    <strong>📊 Measurements Captured:</strong> Height, Chest, Waist, Hips, Shoulders, Arms, Legs, Torso + 15 additional body points for precise fitting
                  </p>
                </div>
              </div>
              
              {/* Height Input for Reference Scaling */}
              <div className="mb-8 max-w-md mx-auto">
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Your Height (for accurate measurements)
                </label>
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <input
                      type="number"
                      value={userHeight}
                      onChange={(e) => setUserHeight(e.target.value)}
                      placeholder="170"
                      min="120"
                      max="220"
                      className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                    <span className="text-white/60 text-xs mt-1 block">cm</span>
                  </div>
                  <div className="text-white/60 text-sm">
                    {userHeight && !isNaN(userHeight) && userHeight > 0 ? 
                      Math.floor(parseFloat(userHeight) / 30.48) + "'" + Math.round((parseFloat(userHeight) / 2.54) % 12) + '"' : 
                      '5\'7"'
                    }
                  </div>
                </div>
                <p className="text-white/50 text-xs mt-2">
                  Providing your height helps us extract more accurate body measurements from your photo
                </p>
              </div>
              
              <div className="relative inline-block">
                {isCameraActive ? (
                  <div className="space-y-4">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full max-w-md mx-auto rounded-lg shadow-lg"
                      style={{ 
                        minHeight: '300px', 
                        backgroundColor: '#1a1a1a',
                        width: '100%',
                        maxWidth: '400px',
                        height: 'auto',
                        objectFit: 'cover'
                      }}
                      onLoadedMetadata={() => {
                        console.log('Video metadata loaded');
                        console.log('Video dimensions:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
                        console.log('Video srcObject:', videoRef.current?.srcObject);
                      }}
                      onError={(e) => {
                        console.error('Video error:', e);
                        console.error('Video element:', videoRef.current);
                      }}
                      onCanPlay={() => {
                        console.log('Video can play');
                        console.log('Video readyState:', videoRef.current?.readyState);
                      }}
                      onPlaying={() => {
                        console.log('Video is playing');
                        console.log('Video paused:', videoRef.current?.paused);
                      }}
                      onLoadStart={() => console.log('Video load start')}
                      onWaiting={() => console.log('Video waiting')}
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    <div className="flex space-x-4 justify-center">
                      <button
                        onClick={capturePhoto}
                        className="btn-primary flex items-center"
                        disabled={!userHeight || isNaN(userHeight) || parseFloat(userHeight) < 120 || parseFloat(userHeight) > 220}
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
                        disabled={!userHeight || isNaN(userHeight) || parseFloat(userHeight) < 120 || parseFloat(userHeight) > 220}
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
                            setTimeout(() => {
                              extractMeasurementsFromImage();
                              setStep(2);
                            }, 500);
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
                    Height: {measurements.height?.toFixed(2)}" ({userHeight || '170'} cm), 
                    Chest: {measurements.chest?.toFixed(2)}", 
                    Waist: {measurements.waist?.toFixed(2)}"
                  </div>
                  <div className="text-green-300/70 text-xs mt-2">
                    Using height reference: {userHeight || '170'} cm for accurate scaling
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 1: User Photo Upload (Upload Mode) */}
        {step === 1.5 && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-white mb-4">Review Your Measurements</h2>
              <p className="text-gray-300 mb-6">
                We've estimated your measurements from your photo. Please review and adjust if needed before proceeding.
              </p>
            </div>

            {userImagePreview && (
              <div className="flex justify-center mb-6">
                <img 
                  src={userImagePreview} 
                  alt="Captured photo" 
                  className="max-w-xs rounded-lg shadow-lg"
                />
              </div>
            )}

            <div className="bg-gray-800/50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Estimated Measurements</h3>
              <div className="max-h-96 overflow-y-auto space-y-6">
                {/* Basic Info Section */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white border-b border-gray-600 pb-2">Basic Information</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Height (inches)</label>
                      <input
                        type="number"
                        value={measurements.height?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, height: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="48"
                        max="84"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Weight (lbs)</label>
                      <input
                        type="number"
                        value={measurements.weight?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, weight: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="80"
                        max="400"
                      />
                    </div>
                  </div>
                </div>

                {/* Head/Neck Section */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white border-b border-gray-600 pb-2">Head & Neck</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Head Circumference (inches)</label>
                      <input
                        type="number"
                        value={measurements.head_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, head_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="20"
                        max="26"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Neck Circumference (inches)</label>
                      <input
                        type="number"
                        value={measurements.neck_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, neck_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="12"
                        max="20"
                      />
                    </div>
                  </div>
                </div>

                {/* Upper Body Section */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white border-b border-gray-600 pb-2">Upper Body</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Chest (inches)</label>
                      <input
                        type="number"
                        value={measurements.chest_circumference?.toFixed(2) || measurements.chest?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, chest_circumference: parseFloat(e.target.value), chest: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="28"
                        max="60"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Bust (inches)</label>
                      <input
                        type="number"
                        value={measurements.bust_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, bust_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="28"
                        max="60"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Underbust (inches)</label>
                      <input
                        type="number"
                        value={measurements.underbust_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, underbust_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="26"
                        max="50"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Waist (inches)</label>
                      <input
                        type="number"
                        value={measurements.waist_circumference?.toFixed(2) || measurements.waist?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, waist_circumference: parseFloat(e.target.value), waist: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="24"
                        max="50"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Shoulder Width (inches)</label>
                      <input
                        type="number"
                        value={measurements.shoulder_width?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, shoulder_width: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="14"
                        max="24"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Arm Length (inches)</label>
                      <input
                        type="number"
                        value={measurements.arm_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, arm_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="18"
                        max="30"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Forearm Length (inches)</label>
                      <input
                        type="number"
                        value={measurements.forearm_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, forearm_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="8"
                        max="14"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Bicep (inches)</label>
                      <input
                        type="number"
                        value={measurements.bicep_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, bicep_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="8"
                        max="20"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Wrist (inches)</label>
                      <input
                        type="number"
                        value={measurements.wrist_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, wrist_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="5"
                        max="9"
                      />
                    </div>
                  </div>
                </div>

                {/* Lower Body Section */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white border-b border-gray-600 pb-2">Lower Body</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Hips (inches)</label>
                      <input
                        type="number"
                        value={measurements.hip_circumference?.toFixed(2) || measurements.hips?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, hip_circumference: parseFloat(e.target.value), hips: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="28"
                        max="55"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Thigh (inches)</label>
                      <input
                        type="number"
                        value={measurements.thigh_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, thigh_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="16"
                        max="32"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Knee (inches)</label>
                      <input
                        type="number"
                        value={measurements.knee_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, knee_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="12"
                        max="20"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Calf (inches)</label>
                      <input
                        type="number"
                        value={measurements.calf_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, calf_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="10"
                        max="20"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Ankle (inches)</label>
                      <input
                        type="number"
                        value={measurements.ankle_circumference?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, ankle_circumference: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="7"
                        max="12"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Inseam (inches)</label>
                      <input
                        type="number"
                        value={measurements.inseam_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, inseam_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="26"
                        max="38"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Outseam (inches)</label>
                      <input
                        type="number"
                        value={measurements.outseam_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, outseam_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="36"
                        max="48"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Rise (inches)</label>
                      <input
                        type="number"
                        value={measurements.rise_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, rise_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.25"
                        min="8"
                        max="15"
                      />
                    </div>
                  </div>
                </div>

                {/* Torso Section */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white border-b border-gray-600 pb-2">Torso</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Torso Length (inches)</label>
                      <input
                        type="number"
                        value={measurements.torso_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, torso_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="20"
                        max="32"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Back Length (inches)</label>
                      <input
                        type="number"
                        value={measurements.back_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, back_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="14"
                        max="22"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Sleeve Length (inches)</label>
                      <input
                        type="number"
                        value={measurements.sleeve_length?.toFixed(2) || ''}
                        onChange={(e) => setMeasurements({...measurements, sleeve_length: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                        step="0.5"
                        min="20"
                        max="30"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={() => setStep(1)}
                className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200"
              >
                Retake Photo
              </button>
              <button
                onClick={() => {
                  saveMeasurementsToBackend(measurements);
                  setStep(2);
                }}
                className="flex-1 btn-primary"
              >
                Continue with These Measurements
              </button>
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="max-w-2xl mx-auto">
            <div className="card text-center">
              <h2 className="text-2xl font-bold text-white mb-6">Upload Your Full Body Photo</h2>
              <div className="mb-6 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg max-w-2xl mx-auto">
                <h3 className="text-orange-200 font-medium mb-3">📋 Photo Requirements for Accurate Measurements</h3>
                <div className="text-sm text-orange-100/80 space-y-2">
                  <p><strong className="text-orange-200">Essential:</strong> Full body visible from head to feet</p>
                  <p><strong className="text-orange-200">Pose:</strong> Stand straight, arms slightly away from body</p>
                  <p><strong className="text-orange-200">Clothing:</strong> Form-fitting clothes (avoid loose/baggy items)</p>
                  <p><strong className="text-orange-200">Background:</strong> Plain wall or contrasting background</p>
                  <p><strong className="text-orange-200">Quality:</strong> Clear, well-lit, no blur or shadows</p>
                </div>
              </div>
              
              {!userImagePreview ? (
                <div>
                  <div 
                    className="upload-area"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Camera className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                    <p className="text-white/80 text-lg mb-2">Click to upload your full-body photo</p>
                    <p className="text-white/60 text-sm">JPG, PNG, HEIC supported (Max 10MB)</p>
                    <p className="text-white/50 text-xs mt-2">Ensure full body is visible for accurate measurements</p>
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
                    onLoad={() => console.log('User image preview loaded successfully')}
                    onError={(e) => {
                      console.error('User image preview failed to load:', e);
                      console.error('Image src:', userImagePreview?.substring(0, 100));
                    }}
                    style={{ maxWidth: '100%', height: 'auto' }}
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

              {(user.measurements || measurements || user.captured_image) && (
                <div className="mt-6 p-4 bg-green-500/20 rounded-lg">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-200 font-medium">
                      {user.captured_image ? "Saved photo available" : "Measurements available"}
                    </span>
                  </div>
                  {(user.measurements || measurements) && (
                    <label className="flex items-center justify-center space-x-3 mb-4">
                      <input
                        type="checkbox"
                        checked={useStoredMeasurements}
                        onChange={(e) => setUseStoredMeasurements(e.target.checked)}
                        className="rounded border-green-400 text-green-600 focus:ring-green-500"
                      />
                      <span className="text-green-200">Use my measurements for better accuracy</span>
                    </label>
                  )}
                  
                  <div className="text-center">
                    <button
                      onClick={() => setStep(2)}
                      className="btn-primary"
                    >
                      {user.captured_image ? "Skip Photo - Use Saved Image" : "Skip Photo - Use Measurements Only"}
                    </button>
                  </div>
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
              
              {user.captured_image && (
                <div className="mb-6 p-4 bg-blue-500/20 rounded-lg">
                  <div className="flex items-center justify-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-blue-400" />
                    <span className="text-blue-200 font-medium">Using your saved photo</span>
                  </div>
                  <p className="text-blue-200/80 text-sm">We'll use your previously captured image for the virtual try-on</p>
                </div>
              )}
              
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
                  <VirtualTryOnResult apiResponse={{result: tryonResult.result_image_base64}} />
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
                              <div>Photo Analysis &amp; Segmentation</div>
                              <div>Pose Estimation &amp; Mapping</div>
                              <div>Identity Preservation</div>
                              <div>Garment Integration</div>
                              <div>Realistic Blending</div>
                              <div>Quality Enhancement</div>
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

              {processingType === 'default' && (
                <div className="bg-yellow-500/20 rounded-lg p-4 mt-4 border border-yellow-500/50">
                  <div className="flex items-start space-x-3">
                    <div className="text-yellow-400 mt-1">💡</div>
                    <div>
                      <h4 className="text-yellow-200 font-semibold mb-2">Want Better Results?</h4>
                      <p className="text-yellow-200/90 text-sm mb-3">
                        For more realistic virtual try-on with advanced identity preservation, try our Premium processing powered by fal.ai FASHN v1.6.
                      </p>
                      <button
                        onClick={() => {
                          setProcessingType('premium');
                          setStep(3);
                        }}
                        className="btn-primary text-sm"
                      >
                        Try Premium Processing
                      </button>
                    </div>
                  </div>
                </div>
              )}

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

        {/* Photo Guide Modal */}
        <PhotoGuideModal 
          isOpen={showPhotoGuide} 
          onClose={() => setShowPhotoGuide(false)} 
        />
      </div>
    </div>
  );
};

export default VirtualTryOn;
