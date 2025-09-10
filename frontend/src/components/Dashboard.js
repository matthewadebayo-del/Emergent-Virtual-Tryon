import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  User, 
  LogOut, 
  Camera, 
  Shirt, 
  History, 
  Settings,
  TrendingUp,
  Zap,
  Target,
  ArrowRight,
  AlertCircle
} from 'lucide-react';
import axios from 'axios';

const Dashboard = ({ user, onLogout }) => {
  const [measurements, setMeasurements] = useState(null);
  const [tryonHistory, setTryonHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showMeasurements, setShowMeasurements] = useState(false);
  const [isFirstTime, setIsFirstTime] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const historyResponse = await axios.get('/tryon-history');
      setTryonHistory(historyResponse.data);
      
      if (user.measurements) {
        setMeasurements(user.measurements);
      } else {
        // First time user - no measurements stored
        setIsFirstTime(true);
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMeasurementSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    const heightFeet = parseFloat(formData.get('height_feet')) || 0;
    const heightInches = parseFloat(formData.get('height_inches')) || 0;
    const totalHeightInches = (heightFeet * 12) + heightInches;
    
    const measurementData = {
      height: totalHeightInches,
      weight: parseFloat(formData.get('weight')),
      
      // Head/neck measurements
      head_circumference: parseFloat(formData.get('head_circumference')) || null,
      neck_circumference: parseFloat(formData.get('neck_circumference')) || null,
      
      // Upper body measurements
      shoulder_width: parseFloat(formData.get('shoulder_width')),
      chest: parseFloat(formData.get('chest_circumference')),
      chest_circumference: parseFloat(formData.get('chest_circumference')),
      bust_circumference: parseFloat(formData.get('bust_circumference')) || null,
      underbust_circumference: parseFloat(formData.get('underbust_circumference')) || null,
      waist: parseFloat(formData.get('waist_circumference')),
      waist_circumference: parseFloat(formData.get('waist_circumference')),
      arm_length: parseFloat(formData.get('arm_length')) || null,
      forearm_length: parseFloat(formData.get('forearm_length')) || null,
      bicep_circumference: parseFloat(formData.get('bicep_circumference')) || null,
      wrist_circumference: parseFloat(formData.get('wrist_circumference')) || null,
      
      // Lower body measurements
      hips: parseFloat(formData.get('hip_circumference')),
      hip_circumference: parseFloat(formData.get('hip_circumference')),
      thigh_circumference: parseFloat(formData.get('thigh_circumference')) || null,
      knee_circumference: parseFloat(formData.get('knee_circumference')) || null,
      calf_circumference: parseFloat(formData.get('calf_circumference')) || null,
      ankle_circumference: parseFloat(formData.get('ankle_circumference')) || null,
      inseam_length: parseFloat(formData.get('inseam_length')) || null,
      outseam_length: parseFloat(formData.get('outseam_length')) || null,
      rise_length: parseFloat(formData.get('rise_length')) || null,
      
      // Torso measurements
      torso_length: parseFloat(formData.get('torso_length')) || null,
      back_length: parseFloat(formData.get('back_length')) || null,
      sleeve_length: parseFloat(formData.get('sleeve_length')) || null
    };

    try {
      await axios.post('/measurements', measurementData);
      setMeasurements(measurementData);
      setShowMeasurements(false);
      setIsFirstTime(false);
      alert('Measurements saved successfully!');
    } catch (error) {
      console.error('Failed to save measurements:', error);
      alert('Failed to save measurements. Please try again.');
    }
  };

  const handleProfileReset = async () => {
    if (window.confirm('Are you sure you want to reset your profile? This will delete all measurements and captured images.')) {
      try {
        await axios.delete('/profile/reset');
        setMeasurements(null);
        setIsFirstTime(true);
        alert('Profile reset successfully!');
      } catch (error) {
        console.error('Failed to reset profile:', error);
        alert('Failed to reset profile. Please try again.');
      }
    }
  };

  const startCameraCapture = () => {
    navigate('/tryon?mode=camera-first');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-300"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Navigation */}
      <nav className="glass-dark sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <Shirt className="w-8 h-8 text-purple-400" />
              <span className="text-2xl font-bold gradient-text">VirtualFit</span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-white/80">Welcome, {user.full_name}</span>
              <button
                onClick={onLogout}
                className="flex items-center space-x-2 text-white/60 hover:text-white transition-colors"
              >
                <LogOut className="w-5 h-5" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            {isFirstTime ? "Welcome to VirtualFit!" : "Dashboard"}
          </h1>
          <p className="text-white/80">
            {isFirstTime 
              ? "Let's get you started with your first virtual try-on experience!"
              : "Ready to try on some clothes?"
            }
          </p>
        </div>

        {/* First Time User Onboarding */}
        {isFirstTime && (
          <div className="mb-8">
            <div className="card bg-gradient-to-r from-purple-600/20 to-blue-600/20 border-2 border-purple-400/30">
              <div className="flex items-start space-x-4">
                <AlertCircle className="w-6 h-6 text-purple-400 mt-1" />
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-white mb-2">Get Started with Camera Capture</h3>
                  <p className="text-white/80 mb-6">
                    For the best virtual try-on experience, we'll help you take a full-body photo using your camera. 
                    This allows us to capture your measurements accurately and create an avatar that looks like you.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4">
                    <button
                      onClick={startCameraCapture}
                      className="btn-primary flex items-center justify-center"
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Start Camera Capture
                    </button>
                    <button
                      onClick={() => setShowMeasurements(true)}
                      className="btn-secondary flex items-center justify-center"
                    >
                      <Target className="w-5 h-5 mr-2" />
                      Enter Measurements Manually
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {(user.captured_image || (measurements && Object.keys(measurements).length > 0)) && (
            <Link to="/tryon" className="card hover-lift">
              <div className="flex items-center space-x-4">
                <div className="p-3 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500">
                  <Shirt className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">Virtual Try-On</h3>
                  <p className="text-white/70">
                    {user.captured_image && measurements 
                      ? "Try clothes with your saved data" 
                      : user.captured_image 
                        ? "Try clothes with your saved photo"
                        : "Try clothes with your measurements"
                    }
                  </p>
                </div>
              </div>
              <ArrowRight className="w-5 h-5 text-purple-400 ml-auto" />
            </Link>
          )}
          
          <button
            onClick={startCameraCapture}
            className="card hover-lift text-left"
          >
            <div className="flex items-center space-x-4">
              <div className="p-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500">
                <Camera className="w-8 h-8 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">Camera Try-On</h3>
                <p className="text-white/70">Capture photo with camera</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-purple-400 ml-auto" />
          </button>

          <Link to="/tryon" className="card hover-lift">
            <div className="flex items-center space-x-4">
              <div className="p-3 rounded-full bg-gradient-to-r from-green-500 to-teal-500">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">Upload & Try-On</h3>
                <p className="text-white/70">Upload existing photo</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-purple-400 ml-auto" />
          </Link>

          <div className="space-y-2">
            <button
              onClick={() => setShowMeasurements(true)}
              className="card hover-lift cursor-pointer text-left w-full"
            >
              <div className="flex items-center space-x-4">
                <div className="p-3 rounded-full bg-gradient-to-r from-orange-500 to-red-500">
                  <Target className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">
                    {measurements ? 'Update' : 'Add'} Measurements
                  </h3>
                  <p className="text-white/70">
                    {measurements ? 'Update your body measurements' : 'Add your body measurements'}
                  </p>
                </div>
              </div>
              <ArrowRight className="w-5 h-5 text-purple-400 ml-auto" />
            </button>
            {measurements && (
              <button
                onClick={handleProfileReset}
                className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition duration-200 text-sm"
              >
                Reset Profile
              </button>
            )}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="card-dark text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">{tryonHistory.length}</div>
            <div className="text-white/70">Try-On Sessions</div>
          </div>
          <div className="card-dark text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {measurements ? '✓' : '0'}
            </div>
            <div className="text-white/70">Measurements Saved</div>
          </div>
          <div className="card-dark text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">95%</div>
            <div className="text-white/70">Accuracy Rate</div>
          </div>
          <div className="card-dark text-center">
            <div className="text-3xl font-bold text-yellow-400 mb-2">2.3s</div>
            <div className="text-white/70">Avg Processing Time</div>
          </div>
        </div>

        {/* Current Measurements */}
        {measurements && (
          <div className="card mb-8">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2" />
              Your Measurements
            </h3>
            <div className="grid md:grid-cols-3 gap-4 text-white/80">
              <div>Height: {measurements.height.toFixed(2)}" ({(measurements.height * 2.54).toFixed(2)} cm)</div>
              <div>Weight: {measurements.weight.toFixed(2)} lbs ({(measurements.weight / 2.205).toFixed(2)} kg)</div>
              <div>Chest: {measurements.chest.toFixed(2)}" ({(measurements.chest * 2.54).toFixed(2)} cm)</div>
              <div>Waist: {measurements.waist.toFixed(2)}" ({(measurements.waist * 2.54).toFixed(2)} cm)</div>
              <div>Hips: {measurements.hips.toFixed(2)}" ({(measurements.hips * 2.54).toFixed(2)} cm)</div>
              <div>Shoulder Width: {measurements.shoulder_width.toFixed(2)}" ({(measurements.shoulder_width * 2.54).toFixed(2)} cm)</div>
            </div>
          </div>
        )}

        {/* Recent Try-Ons */}
        {tryonHistory.length > 0 && (
          <div className="card">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <History className="w-5 h-5 mr-2" />
              Recent Try-Ons
            </h3>
            <div className="space-y-4">
              {tryonHistory.slice(0, 5).map((session, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                  <div>
                    <div className="text-white font-medium">
                      Try-On Session #{index + 1}
                    </div>
                    <div className="text-white/60 text-sm">
                      Size Recommendation: {session.size_recommendation}
                    </div>
                  </div>
                  <div className="text-white/60 text-sm">
                    {new Date(session.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Measurements Modal */}
      {showMeasurements && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="card max-w-md w-full">
            <h3 className="text-xl font-semibold text-white mb-4">
              {measurements ? 'Update' : 'Add'} Your Measurements
            </h3>
            <form onSubmit={handleMeasurementSubmit} className="space-y-6 max-h-96 overflow-y-auto">
              {/* Basic Info Section */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-white border-b border-white/20 pb-2">Basic Information</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Height (feet & inches)</label>
                    <div className="flex gap-2">
                      <input type="number" name="height_feet" defaultValue={measurements?.height ? Math.floor(measurements.height / 12) : ''} className="input-field flex-1" placeholder="5" min="3" max="8" required />
                      <span className="text-white/60 self-center">ft</span>
                      <input type="number" name="height_inches" defaultValue={measurements?.height ? Math.round(measurements.height % 12) : ''} className="input-field flex-1" placeholder="8" min="0" max="11" required />
                      <span className="text-white/60 self-center">in</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Weight (lbs)</label>
                    <input type="number" name="weight" defaultValue={measurements?.weight || ''} className="input-field" placeholder="150" min="80" max="400" step="0.5" required />
                  </div>
                </div>
              </div>

              {/* Head/Neck Section */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-white border-b border-white/20 pb-2">Head & Neck</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Head Circumference (inches)</label>
                    <input type="number" name="head_circumference" defaultValue={measurements?.head_circumference || ''} className="input-field" placeholder="22" min="20" max="26" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Neck Circumference (inches)</label>
                    <input type="number" name="neck_circumference" defaultValue={measurements?.neck_circumference || ''} className="input-field" placeholder="15" min="12" max="20" step="0.25" />
                  </div>
                </div>
              </div>

              {/* Upper Body Section */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-white border-b border-white/20 pb-2">Upper Body</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Chest (inches)</label>
                    <input type="number" name="chest_circumference" defaultValue={measurements?.chest_circumference || measurements?.chest || ''} className="input-field" placeholder="36" min="28" max="60" step="0.5" required />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Bust (inches)</label>
                    <input type="number" name="bust_circumference" defaultValue={measurements?.bust_circumference || ''} className="input-field" placeholder="36" min="28" max="60" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Underbust (inches)</label>
                    <input type="number" name="underbust_circumference" defaultValue={measurements?.underbust_circumference || ''} className="input-field" placeholder="32" min="26" max="50" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Waist (inches)</label>
                    <input type="number" name="waist_circumference" defaultValue={measurements?.waist_circumference || measurements?.waist || ''} className="input-field" placeholder="32" min="24" max="50" step="0.5" required />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Shoulder Width (inches)</label>
                    <input type="number" name="shoulder_width" defaultValue={measurements?.shoulder_width || ''} className="input-field" placeholder="18" min="14" max="24" step="0.5" required />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Arm Length (inches)</label>
                    <input type="number" name="arm_length" defaultValue={measurements?.arm_length || ''} className="input-field" placeholder="24" min="18" max="30" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Forearm Length (inches)</label>
                    <input type="number" name="forearm_length" defaultValue={measurements?.forearm_length || ''} className="input-field" placeholder="11" min="8" max="14" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Bicep (inches)</label>
                    <input type="number" name="bicep_circumference" defaultValue={measurements?.bicep_circumference || ''} className="input-field" placeholder="12" min="8" max="20" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Wrist (inches)</label>
                    <input type="number" name="wrist_circumference" defaultValue={measurements?.wrist_circumference || ''} className="input-field" placeholder="6.5" min="5" max="9" step="0.25" />
                  </div>
                </div>
              </div>

              {/* Lower Body Section */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-white border-b border-white/20 pb-2">Lower Body</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Hips (inches)</label>
                    <input type="number" name="hip_circumference" defaultValue={measurements?.hip_circumference || measurements?.hips || ''} className="input-field" placeholder="38" min="28" max="55" step="0.5" required />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Thigh (inches)</label>
                    <input type="number" name="thigh_circumference" defaultValue={measurements?.thigh_circumference || ''} className="input-field" placeholder="22" min="16" max="32" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Knee (inches)</label>
                    <input type="number" name="knee_circumference" defaultValue={measurements?.knee_circumference || ''} className="input-field" placeholder="15" min="12" max="20" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Calf (inches)</label>
                    <input type="number" name="calf_circumference" defaultValue={measurements?.calf_circumference || ''} className="input-field" placeholder="14" min="10" max="20" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Ankle (inches)</label>
                    <input type="number" name="ankle_circumference" defaultValue={measurements?.ankle_circumference || ''} className="input-field" placeholder="8.5" min="7" max="12" step="0.25" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Inseam (inches)</label>
                    <input type="number" name="inseam_length" defaultValue={measurements?.inseam_length || ''} className="input-field" placeholder="32" min="26" max="38" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Outseam (inches)</label>
                    <input type="number" name="outseam_length" defaultValue={measurements?.outseam_length || ''} className="input-field" placeholder="42" min="36" max="48" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Rise (inches)</label>
                    <input type="number" name="rise_length" defaultValue={measurements?.rise_length || ''} className="input-field" placeholder="11" min="8" max="15" step="0.25" />
                  </div>
                </div>
              </div>

              {/* Torso Section */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-white border-b border-white/20 pb-2">Torso</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Torso Length (inches)</label>
                    <input type="number" name="torso_length" defaultValue={measurements?.torso_length || ''} className="input-field" placeholder="25" min="20" max="32" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Back Length (inches)</label>
                    <input type="number" name="back_length" defaultValue={measurements?.back_length || ''} className="input-field" placeholder="17" min="14" max="22" step="0.5" />
                  </div>
                  <div>
                    <label className="block text-white/80 text-sm font-medium mb-2">Sleeve Length (inches)</label>
                    <input type="number" name="sleeve_length" defaultValue={measurements?.sleeve_length || ''} className="input-field" placeholder="25" min="20" max="30" step="0.5" />
                  </div>
                </div>
              </div>

              <div className="flex space-x-4 pt-4 border-t border-white/20">
                <button type="submit" className="btn-primary flex-1">Save Measurements</button>
                <button type="button" onClick={() => setShowMeasurements(false)} className="btn-secondary flex-1">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
