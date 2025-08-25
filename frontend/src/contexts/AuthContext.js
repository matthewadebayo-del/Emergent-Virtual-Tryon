import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('token'));

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  // Configure axios and check auth on token change
  useEffect(() => {
    const initAuth = async () => {
      if (token) {
        try {
          // Set the authorization header
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // Verify token by fetching user profile
          const response = await axios.get(`${API}/profile`);
          setUser(response.data);
          console.log('User authenticated successfully:', response.data.email);
        } catch (error) {
          console.error('Auth check failed:', error);
          // Clear invalid token
          localStorage.removeItem('token');
          setToken(null);
          delete axios.defaults.headers.common['Authorization'];
        }
      } else {
        // No token - clear auth headers
        delete axios.defaults.headers.common['Authorization'];
      }
      setLoading(false);
    };

    initAuth();
  }, [API, token]);

  const login = async (email, password) => {
    try {
      const response = await axios.post(`${API}/login`, {
        email,
        password
      });

      const { access_token, user: userData } = response.data;
      
      setUser(userData);
      setToken(access_token);
      localStorage.setItem('token', access_token);
      
      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      return { 
        success: false, 
        message: error.response?.data?.detail || 'Login failed' 
      };
    }
  };

  const register = async (email, password, fullName) => {
    try {
      const response = await axios.post(`${API}/register`, {
        email,
        password,
        full_name: fullName
      });

      const { access_token, user: userData } = response.data;
      
      setUser(userData);
      setToken(access_token);
      localStorage.setItem('token', access_token);
      
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error);
      return { 
        success: false, 
        message: error.response?.data?.detail || 'Registration failed' 
      };
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
  };

  const resetPassword = async (email) => {
    try {
      await axios.post(`${API}/reset-password`, { email });
      return { success: true };
    } catch (error) {
      console.error('Password reset error:', error);
      return { 
        success: false, 
        message: error.response?.data?.detail || 'Password reset failed' 
      };
    }
  };

  const updateUser = (userData) => {
    console.log('Updating user data in context:', userData);
    setUser(userData);
  };

  const value = {
    user,
    login,
    register,
    logout,
    resetPassword,
    updateUser,
    loading,
    token
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};