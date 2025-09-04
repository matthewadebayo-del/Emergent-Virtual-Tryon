import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Shirt, Mail, Lock, Eye, EyeOff, LogIn } from 'lucide-react';
import axios from 'axios';

const LoginPage = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showResetForm, setShowResetForm] = useState(false);
  const [resetEmail, setResetEmail] = useState('');
  const [resetMessage, setResetMessage] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await axios.post('/login', formData);
      const { access_token } = response.data;
      
      // Get user profile
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      const profileResponse = await axios.get('/profile');
      
      onLogin(access_token, profileResponse.data);
    } catch (error) {
      setError(error.response?.data?.detail || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordReset = async (e) => {
    e.preventDefault();
    setResetMessage('');

    try {
      await axios.post('/reset-password', { email: resetEmail });
      setResetMessage('Password reset instructions have been sent to your email.');
    } catch (error) {
      setResetMessage('Error sending reset email. Please try again.');
    }
  };

  if (showResetForm) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center px-4">
        <div className="max-w-md w-full">
          <div className="text-center mb-8">
            <Link to="/" className="inline-flex items-center space-x-2 mb-4">
              <Shirt className="w-10 h-10 text-purple-400" />
              <span className="text-3xl font-bold gradient-text">VirtualFit</span>
            </Link>
            <h2 className="text-2xl font-bold text-white mb-2">Reset Password</h2>
            <p className="text-white/70">Enter your email to receive reset instructions</p>
          </div>

          <div className="card">
            <form onSubmit={handlePasswordReset} className="space-y-6">
              {resetMessage && (
                <div className={`rounded-lg p-3 text-sm ${
                  resetMessage.includes('sent') 
                    ? 'bg-green-500/20 border border-green-500/50 text-green-200'
                    : 'bg-red-500/20 border border-red-500/50 text-red-200'
                }`}>
                  {resetMessage}
                </div>
              )}

              <div>
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Email Address
                </label>
                <div className="input-field-container">
                  <Mail className="input-icon w-5 h-5" />
                  <input
                    type="email"
                    value={resetEmail}
                    onChange={(e) => setResetEmail(e.target.value)}
                    className="input-field"
                    placeholder="Enter your email"
                    required
                  />
                </div>
              </div>

              <button type="submit" className="w-full btn-primary">
                Send Reset Instructions
              </button>
            </form>

            <div className="mt-6 text-center">
              <button
                onClick={() => setShowResetForm(false)}
                className="text-purple-300 hover:text-purple-200 font-medium"
              >
                Back to Login
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center space-x-2 mb-4">
            <Shirt className="w-10 h-10 text-purple-400" />
            <span className="text-3xl font-bold gradient-text">VirtualFit</span>
          </Link>
          <h2 className="text-2xl font-bold text-white mb-2">Welcome Back</h2>
          <p className="text-white/70">Sign in to continue your virtual try-on experience</p>
        </div>

        {/* Login Form */}
        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-3 text-red-200 text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-white/80 text-sm font-medium mb-2">
                Email Address
              </label>
              <div className="input-field-container">
                <Mail className="input-icon w-5 h-5" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="input-field"
                  placeholder="Enter your email"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-white/80 text-sm font-medium mb-2">
                Password
              </label>
              <div className="input-field-container">
                <Lock className="input-icon w-5 h-5" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="input-field pr-12"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/50 hover:text-white/70"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input type="checkbox" className="rounded border-white/30 text-purple-600 focus:ring-purple-500" />
                <span className="ml-2 text-sm text-white/70">Remember me</span>
              </label>
              <button
                type="button"
                onClick={() => setShowResetForm(true)}
                className="text-sm text-purple-300 hover:text-purple-200"
              >
                Forgot password?
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary flex items-center justify-center"
            >
              {loading ? (
                <div className="spinner mr-2"></div>
              ) : (
                <LogIn className="w-5 h-5 mr-2" />
              )}
              {loading ? 'Signing In...' : 'Sign In'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-white/70">
              Don't have an account?{' '}
              <Link to="/register" className="text-purple-300 hover:text-purple-200 font-medium">
                Sign up
              </Link>
            </p>
          </div>
        </div>

        {/* Demo Access */}
        <div className="mt-6 text-center">
          <p className="text-white/50 text-sm">
            Demo: Use email "demo@virtualfit.com" and password "demo123"
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
