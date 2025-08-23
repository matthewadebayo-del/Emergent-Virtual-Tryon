import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { PlayCircle, Camera, Upload, Zap, Shield, Star } from 'lucide-react';

const LandingPage = () => {
  const [showDemoModal, setShowDemoModal] = useState(false);

  const features = [
    {
      icon: <Camera className="w-8 h-8 text-purple-400" />,
      title: "Capture or Upload Your Photo",
      description: "Take a photo with your camera or upload an existing image to get started with your virtual try-on experience."
    },
    {
      icon: <Zap className="w-8 h-8 text-purple-400" />,
      title: "AI-Powered Fitting",
      description: "Our advanced AI analyzes your body measurements and creates a perfect virtual fit that preserves your exact appearance."
    },
    {
      icon: <Shield className="w-8 h-8 text-purple-400" />,
      title: "Hybrid Technology",
      description: "Choose between our cost-effective hybrid 3D approach or premium fal.ai integration for the highest quality results."
    }
  ];

  const DemoModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-2xl font-bold text-white">How It Works - Demo</h3>
            <button
              onClick={() => setShowDemoModal(false)}
              className="text-gray-400 hover:text-white text-2xl"
            >
              ×
            </button>
          </div>
          
          <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center mb-6">
            <div className="text-center">
              <PlayCircle className="w-16 h-16 text-purple-400 mx-auto mb-4" />
              <p className="text-gray-300">Demo Video Coming Soon</p>
              <p className="text-sm text-gray-500 mt-2">
                Experience the future of virtual try-on technology
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-purple-900 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <span className="text-purple-200 font-bold">1</span>
              </div>
              <h4 className="text-white font-semibold mb-2">Upload Photo</h4>
              <p className="text-gray-400 text-sm">Take or upload a full-body photo</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-900 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <span className="text-purple-200 font-bold">2</span>
              </div>
              <h4 className="text-white font-semibold mb-2">Select Clothing</h4>
              <p className="text-gray-400 text-sm">Choose from our premium catalog</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-900 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                <span className="text-purple-200 font-bold">3</span>
              </div>
              <h4 className="text-white font-semibold mb-2">See Results</h4>
              <p className="text-gray-400 text-sm">Get photorealistic try-on results</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Header */}
      <header className="container mx-auto px-6 py-8">
        <nav className="flex justify-between items-center">
          <div className="text-2xl font-bold text-purple-400">VirtualTryOn</div>
          <div className="space-x-4">
            <Link
              to="/login"
              className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
            >
              Sign In
            </Link>
            <Link
              to="/register"
              className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
            >
              Get Started
            </Link>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
          Virtual Try-On Revolution
        </h1>
        <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
          Experience the future of online shopping with AI-powered virtual try-on technology. 
          See yourself in any outfit before you buy.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/register"
            className="px-8 py-4 bg-purple-600 hover:bg-purple-700 rounded-lg text-lg font-semibold transition-colors"
          >
            Start Trying On
          </Link>
          <button
            onClick={() => setShowDemoModal(true)}
            className="px-8 py-4 border border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white rounded-lg text-lg font-semibold transition-colors flex items-center justify-center gap-2"
          >
            <PlayCircle className="w-5 h-5" />
            Watch Demo
          </button>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-6 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-gray-800 rounded-lg p-8 text-center hover:bg-gray-700 transition-colors">
              <div className="mb-6">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-4">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Technology Section */}
      <section className="container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-6">Advanced Technology Stack</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Choose between our innovative hybrid 3D approach or premium fal.ai integration
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-12">
          {/* Hybrid Approach */}
          <div className="bg-gray-800 rounded-lg p-8">
            <div className="text-center mb-6">
              <div className="bg-green-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-green-400" />
              </div>
              <h3 className="text-2xl font-bold text-green-400">Hybrid 3D Approach</h3>
              <p className="text-sm text-green-300 mt-2">Default • Cost-Effective</p>
            </div>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-green-400" />
                3D Body Reconstruction (MediaPipe + SMPL)
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-green-400" />
                Physics-Based Garment Fitting
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-green-400" />
                Photorealistic Rendering (Blender)
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-green-400" />
                AI Enhancement (Stable Diffusion)
              </li>
            </ul>
            <div className="mt-6 text-center">
              <span className="text-2xl font-bold text-green-400">$0.02</span>
              <span className="text-gray-400"> per try-on</span>
            </div>
          </div>

          {/* Premium Approach */}
          <div className="bg-gray-800 rounded-lg p-8 border border-purple-400">
            <div className="text-center mb-6">
              <div className="bg-purple-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Shield className="w-8 h-8 text-purple-400" />
              </div>
              <h3 className="text-2xl font-bold text-purple-400">Premium fal.ai</h3>
              <p className="text-sm text-purple-300 mt-2">Premium • Highest Quality</p>
            </div>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-purple-400" />
                FASHN Virtual Try-On API
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-purple-400" />
                Identity-Preserving Synthesis
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-purple-400" />
                Instant High-Quality Results
              </li>
              <li className="flex items-center gap-3">
                <Star className="w-4 h-4 text-purple-400" />
                Professional-Grade Output
              </li>
            </ul>
            <div className="mt-6 text-center">
              <span className="text-2xl font-bold text-purple-400">$0.075</span>
              <span className="text-gray-400"> per try-on</span>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <h2 className="text-4xl font-bold mb-6">Ready to Transform Your Shopping?</h2>
        <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
          Join thousands of users who are already experiencing the future of fashion with our virtual try-on technology.
        </p>
        <Link
          to="/register"
          className="inline-block px-8 py-4 bg-purple-600 hover:bg-purple-700 rounded-lg text-lg font-semibold transition-colors"
        >
          Get Started Now
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-8">
        <div className="container mx-auto px-6 text-center text-gray-400">
          <p>&copy; 2025 VirtualTryOn. Powered by advanced AI technology.</p>
        </div>
      </footer>

      {/* Demo Modal */}
      {showDemoModal && <DemoModal />}
    </div>
  );
};

export default LandingPage;