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
  // Token state with better validation
  const [token, setToken] = useState(() => {
    try {
      const storedToken = localStorage.getItem('token');
      // Only return token if it exists, is not empty, and looks like a JWT
      if (storedToken && storedToken.trim() && storedToken.includes('.')) {
        return storedToken;
      }
      return null;
    } catch (error) {
      console.warn('Error reading token from localStorage:', error);
      return null;
    }
  });

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  // Configure axios and check auth on token change
  useEffect(() => {
    const initAuth = async () => {
      // Only attempt auth check if we have a token
      if (token && token.trim()) {
        try {
          console.log('Checking authentication with existing token...');
          
          // Set the authorization header
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // Verify token by fetching user profile
          const response = await axios.get(`${API}/profile`);
          
          if (response.data && response.data.email) {
            setUser(response.data);
            console.log('User authenticated successfully:', response.data.email);
          } else {
            throw new Error('Invalid user data received');
          }
          
        } catch (error) {
          console.warn('Auth check failed, clearing invalid token:', error.message);
          
          // Clear invalid token and auth state
          localStorage.removeItem('token');
          setToken(null);
          setUser(null);
          delete axios.defaults.headers.common['Authorization'];
        }
      } else {
        // No token - ensure clean state
        console.log('No token found, ensuring clean auth state');
        setUser(null);
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
      
      // Set authorization header immediately
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
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
      
      // Set authorization header immediately
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
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