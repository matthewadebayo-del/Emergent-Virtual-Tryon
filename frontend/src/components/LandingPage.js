import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Camera, Shirt, Zap, Users, Star, ChevronRight, Play, CheckCircle } from 'lucide-react';

const LandingPage = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  
  const heroImages = [
    'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1445205170230-053b83016050?w=800&h=600&fit=crop',
    'https://images.unsplash.com/photo-1472851294608-062f824d29cc?w=800&h=600&fit=crop'
  ];

  const features = [
    {
      icon: <Camera className="w-8 h-8" />,
      title: "Smart Body Capture",
      description: "Advanced AI captures your body measurements from a single photo with 99% accuracy."
    },
    {
      icon: <Shirt className="w-8 h-8" />,
      title: "Virtual Try-On",
      description: "See how clothes look on your body before buying. No more guessing sizes!"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Instant Results",
      description: "Get your virtual try-on results in seconds, not minutes."
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Size Recommendations",
      description: "AI-powered size suggestions based on your exact measurements."
    }
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      rating: 5,
      comment: "This app saved me so much time and money! No more returns due to wrong sizes."
    },
    {
      name: "Mike Chen",
      rating: 5,
      comment: "The virtual try-on is incredibly realistic. I'm amazed by the technology."
    },
    {
      name: "Emma Williams",
      rating: 5,
      comment: "Perfect fit every time. The size recommendations are spot on!"
    }
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % heroImages.length);
    }, 5000);
    return () => clearInterval(timer);
  }, [heroImages.length]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-dark">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <Shirt className="w-8 h-8 text-purple-400" />
              <span className="text-2xl font-bold gradient-text">VirtualFit</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#features" className="text-white/80 hover:text-white transition-colors">Features</a>
              <a href="#how-it-works" className="text-white/80 hover:text-white transition-colors">How it Works</a>
              <a href="#testimonials" className="text-white/80 hover:text-white transition-colors">Reviews</a>
            </div>
            <div className="flex space-x-4">
              <Link to="/login" className="text-purple-300 hover:text-white transition-colors">
                Sign In
              </Link>
              <Link to="/register" className="btn-primary">
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-16 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="animate-fade-in-up">
              <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight">
                Try Before You
                <span className="gradient-text"> Buy</span>
              </h1>
              <p className="text-xl text-white/80 mb-8 leading-relaxed">
                Revolutionary virtual try-on technology that uses AI to show you exactly how clothes will look and fit on your body. Get perfect sizing recommendations every time.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 mb-8">
                <Link to="/register" className="btn-primary text-center">
                  Start Virtual Try-On
                  <ChevronRight className="w-5 h-5 ml-2 inline" />
                </Link>
                <button className="btn-secondary flex items-center justify-center">
                  <Play className="w-5 h-5 mr-2" />
                  Watch Demo
                </button>
              </div>
              <div className="flex items-center space-x-6 text-white/60">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span>No credit card required</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span>Free to start</span>
                </div>
              </div>
            </div>
            
            <div className="animate-slide-in-right">
              <div className="relative carousel-container h-96 lg:h-[500px]">
                {heroImages.map((image, index) => (
                  <img
                    key={index}
                    src={image}
                    alt={`Hero ${index + 1}`}
                    className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ${
                      index === currentSlide ? 'opacity-100' : 'opacity-0'
                    }`}
                  />
                ))}
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
                  {heroImages.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentSlide(index)}
                      className={`w-3 h-3 rounded-full transition-colors ${
                        index === currentSlide ? 'bg-purple-400' : 'bg-white/30'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Why Choose <span className="gradient-text">VirtualFit</span>?
            </h2>
            <p className="text-xl text-white/80 max-w-3xl mx-auto">
              Our cutting-edge AI technology revolutionizes online shopping by eliminating the guesswork from clothing purchases.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="card text-center hover-lift">
                <div className="feature-icon">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                <p className="text-white/70">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-4 bg-black/20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              How It <span className="gradient-text">Works</span>
            </h2>
            <p className="text-xl text-white/80 max-w-3xl mx-auto">
              Get started in just three simple steps and see how clothes look on you before buying.
            </p>
          </div>
          
          <div className="grid lg:grid-cols-3 gap-12">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center text-white text-2xl font-bold">
                1
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Upload Your Photo</h3>
              <p className="text-white/70">
                Take or upload a full-body photo. Our AI will automatically extract your body measurements with incredible precision.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center text-white text-2xl font-bold">
                2
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Choose Your Style</h3>
              <p className="text-white/70">
                Browse our extensive catalog or upload your own clothing item. Select what you'd like to try on virtually.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center text-white text-2xl font-bold">
                3
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">See the Magic</h3>
              <p className="text-white/70">
                Get instant virtual try-on results with perfect size recommendations. Make confident purchase decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              What Our <span className="gradient-text">Users Say</span>
            </h2>
            <p className="text-xl text-white/80 max-w-3xl mx-auto">
              Join thousands of satisfied customers who've transformed their online shopping experience.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="card">
                <div className="flex items-center mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-white/80 mb-4 italic">"{testimonial.comment}"</p>
                <div className="font-semibold text-white">{testimonial.name}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-purple-600 to-blue-600">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            Ready to Transform Your Shopping?
          </h2>
          <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
            Join the future of online fashion. Get started with your first virtual try-on today and never worry about sizing again.
          </p>
          <Link to="/register" className="btn-primary text-xl py-4 px-8">
            Start Your Virtual Try-On Journey
            <ChevronRight className="w-6 h-6 ml-2 inline" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 bg-black/40">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Shirt className="w-8 h-8 text-purple-400" />
                <span className="text-2xl font-bold gradient-text">VirtualFit</span>
              </div>
              <p className="text-white/70">
                Revolutionary virtual try-on technology for the future of fashion.
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-white/70">
                <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-white/70">
                <li><a href="#" className="hover:text-white transition-colors">About</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-white/70">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Privacy</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/20 mt-8 pt-8 text-center text-white/60">
            <p>&copy; 2025 VirtualFit. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;