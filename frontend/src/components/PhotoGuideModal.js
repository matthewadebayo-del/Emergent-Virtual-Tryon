import React from 'react';
import { X, Camera, User, Ruler } from 'lucide-react';

const PhotoGuideModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white flex items-center">
              <Camera className="w-6 h-6 mr-2 text-blue-400" />
              Full Body Measurement Photo Guide
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Correct Position */}
            <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
              <h3 className="text-green-200 font-semibold mb-3 flex items-center">
                ✅ Correct Position
              </h3>
              <div className="space-y-3 text-sm text-green-100/80">
                <div className="flex items-start space-x-2">
                  <User className="w-4 h-4 mt-0.5 text-green-400" />
                  <div>
                    <strong>Body Stance:</strong>
                    <ul className="mt-1 space-y-1 text-xs">
                      <li>• Stand straight, shoulders back</li>
                      <li>• Feet shoulder-width apart</li>
                      <li>• Arms slightly away from body</li>
                      <li>• Face camera directly</li>
                    </ul>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <Ruler className="w-4 h-4 mt-0.5 text-green-400" />
                  <div>
                    <strong>Measurement Points Visible:</strong>
                    <ul className="mt-1 space-y-1 text-xs">
                      <li>• Shoulders clearly defined</li>
                      <li>• Waist narrowest point visible</li>
                      <li>• Hip widest point visible</li>
                      <li>• Full arm and leg length</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Incorrect Position */}
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <h3 className="text-red-200 font-semibold mb-3">
                ❌ Avoid These Mistakes
              </h3>
              <div className="space-y-2 text-sm text-red-100/80">
                <p>• Arms pressed against body</p>
                <p>• Slouching or leaning</p>
                <p>• Partial body in frame</p>
                <p>• Loose, baggy clothing</p>
                <p>• Poor lighting/shadows</p>
                <p>• Cluttered background</p>
                <p>• Too close to camera</p>
                <p>• Angled or side view</p>
              </div>
            </div>
          </div>

          {/* Technical Requirements */}
          <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <h3 className="text-blue-200 font-semibold mb-3">📋 Technical Requirements</h3>
            <div className="grid md:grid-cols-3 gap-4 text-sm text-blue-100/80">
              <div>
                <strong className="text-blue-200">Distance & Framing:</strong>
                <ul className="mt-1 space-y-1 text-xs">
                  <li>• 6-8 feet from camera</li>
                  <li>• Full body head to feet</li>
                  <li>• Camera at chest height</li>
                  <li>• Vertical orientation</li>
                </ul>
              </div>
              <div>
                <strong className="text-blue-200">Lighting & Environment:</strong>
                <ul className="mt-1 space-y-1 text-xs">
                  <li>• Natural daylight preferred</li>
                  <li>• Even lighting, no shadows</li>
                  <li>• Plain background</li>
                  <li>• No backlighting</li>
                </ul>
              </div>
              <div>
                <strong className="text-blue-200">Clothing & Accessories:</strong>
                <ul className="mt-1 space-y-1 text-xs">
                  <li>• Form-fitting clothes</li>
                  <li>• Remove bulky items</li>
                  <li>• Minimal accessories</li>
                  <li>• Contrasting colors</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Measurements Captured */}
          <div className="mt-6 bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
            <h3 className="text-purple-200 font-semibold mb-3">📏 Measurements We'll Extract</h3>
            <div className="grid md:grid-cols-4 gap-3 text-xs text-purple-100/80">
              <div>
                <strong className="text-purple-200">Basic:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Height</li>
                  <li>• Weight (estimated)</li>
                </ul>
              </div>
              <div>
                <strong className="text-purple-200">Upper Body:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Chest/Bust</li>
                  <li>• Waist</li>
                  <li>• Shoulders</li>
                  <li>• Arms</li>
                  <li>• Neck</li>
                </ul>
              </div>
              <div>
                <strong className="text-purple-200">Lower Body:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Hips</li>
                  <li>• Thighs</li>
                  <li>• Inseam</li>
                  <li>• Outseam</li>
                </ul>
              </div>
              <div>
                <strong className="text-purple-200">Detailed:</strong>
                <ul className="mt-1 space-y-1">
                  <li>• Torso length</li>
                  <li>• Sleeve length</li>
                  <li>• Rise</li>
                  <li>• + 10 more</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-center">
            <button
              onClick={onClose}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
            >
              Got it, let's take the photo!
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhotoGuideModal;