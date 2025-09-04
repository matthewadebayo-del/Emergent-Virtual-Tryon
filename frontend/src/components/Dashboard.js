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
      const historyResponse = await axios.get('/api/tryon-history');
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
    const measurementData = {
      height: parseFloat(formData.get('height')),
      weight: parseFloat(formData.get('weight')),
      chest: parseFloat(formData.get('chest')),
      waist: parseFloat(formData.get('waist')),
      hips: parseFloat(formData.get('hips')),
      shoulder_width: parseFloat(formData.get('shoulder_width'))
    };

    try {
      await axios.post('/api/measurements', measurementData);
      setMeasurements(measurementData);
      setShowMeasurements(false);
      setIsFirstTime(false);
      alert('Measurements saved successfully!');
    } catch (error) {
      console.error('Failed to save measurements:', error);
      alert('Failed to save measurements. Please try again.');
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

          <button
            onClick={() => setShowMeasurements(true)}
            className="card hover-lift cursor-pointer text-left"
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
              <div>Height: {measurements.height}" ({(measurements.height * 2.54).toFixed(0)} cm)</div>
              <div>Weight: {measurements.weight} lbs ({(measurements.weight / 2.205).toFixed(0)} kg)</div>
              <div>Chest: {measurements.chest}" ({(measurements.chest * 2.54).toFixed(0)} cm)</div>
              <div>Waist: {measurements.waist}" ({(measurements.waist * 2.54).toFixed(0)} cm)</div>
              <div>Hips: {measurements.hips}" ({(measurements.hips * 2.54).toFixed(0)} cm)</div>
              <div>Shoulder Width: {measurements.shoulder_width}" ({(measurements.shoulder_width * 2.54).toFixed(0)} cm)</div>
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
            <form onSubmit={handleMeasurementSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Height (cm)
                  </label>
                  <input
                    type="number"
                    name="height"
                    defaultValue={measurements?.height || ''}
                    className="input-field"
                    placeholder="170"
                    required
                  />
                </div>
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Weight (kg)
                  </label>
                  <input
                    type="number"
                    name="weight"
                    defaultValue={measurements?.weight || ''}
                    className="input-field"
                    placeholder="70"
                    required
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Chest (cm)
                  </label>
                  <input
                    type="number"
                    name="chest"
                    defaultValue={measurements?.chest || ''}
                    className="input-field"
                    placeholder="90"
                    required
                  />
                </div>
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Waist (cm)
                  </label>
                  <input
                    type="number"
                    name="waist"
                    defaultValue={measurements?.waist || ''}
                    className="input-field"
                    placeholder="75"
                    required
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Hips (cm)
                  </label>
                  <input
                    type="number"
                    name="hips"
                    defaultValue={measurements?.hips || ''}
                    className="input-field"
                    placeholder="95"
                    required
                  />
                </div>
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Shoulder Width (cm)
                  </label>
                  <input
                    type="number"
                    name="shoulder_width"
                    defaultValue={measurements?.shoulder_width || ''}
                    className="input-field"
                    placeholder="45"
                    required
                  />
                </div>
              </div>
              <div className="flex space-x-4">
                <button type="submit" className="btn-primary flex-1">
                  Save Measurements
                </button>
                <button
                  type="button"
                  onClick={() => setShowMeasurements(false)}
                  className="btn-secondary flex-1"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
