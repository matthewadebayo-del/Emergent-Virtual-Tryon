# VirtualFit E-Commerce Integration Guide for Brands

## Overview
VirtualFit provides a comprehensive virtual try-on API that can be seamlessly integrated into existing e-commerce platforms. This guide covers both **Public Mode** (standalone application) and **Brand-Specific Integration** (white-label solutions).

---

## Table of Contents
1. [Integration Modes](#integration-modes)
2. [API Documentation](#api-documentation)
3. [Brand Integration Setup](#brand-integration-setup)
4. [Frontend Widget Integration](#frontend-widget-integration)
5. [Authentication & Security](#authentication--security)
6. [Customization Options](#customization-options)
7. [Webhooks & Callbacks](#webhooks--callbacks)
8. [Error Handling](#error-handling)
9. [Performance & Scaling](#performance--scaling)
10. [Testing & Implementation](#testing--implementation)
11. [Support & Maintenance](#support--maintenance)

---

## Integration Modes

### 1. Public Mode
- Standalone VirtualFit application
- Users access via https://virtual-tryon-app.preview.emergentagent.com
- Brand product catalog integration via API
- Suitable for: Small to medium brands testing virtual try-on

### 2. Brand-Specific Integration
- White-label solution embedded in brand's e-commerce site
- Custom branding and styling
- Seamless user experience within brand's ecosystem
- Suitable for: Large brands, enterprise clients

---

## API Documentation

### Base URL
```
Production: https://virtual-tryon-app.preview.emergentagent.com/api
```

### Authentication
All API requests require a valid JWT token obtained through user authentication.

```bash
# Login to get JWT token
POST /api/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### Core Endpoints

#### 1. User Management
```bash
# Register User
POST /api/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe"
}

# Get User Profile
GET /api/profile
Authorization: Bearer {token}
```

#### 2. Product Catalog
```bash
# Get Products
GET /api/products
Authorization: Bearer {token}

# Add Brand Products (Admin)
POST /api/products
Authorization: Bearer {admin_token}
Content-Type: application/json

{
  "name": "Summer Dress",
  "category": "dresses",
  "sizes": ["XS", "S", "M", "L", "XL"],
  "image_url": "https://brand.com/product-image.jpg",
  "description": "Elegant summer dress",
  "price": 89.99,
  "brand_id": "brand_123",
  "sku": "DRESS_001"
}
```

#### 3. Virtual Try-On
```bash
# Virtual Try-On Request
POST /api/tryon
Authorization: Bearer {token}
Content-Type: application/json

{
  "user_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAA...",
  "product_id": "product_123",
  "use_stored_measurements": true
}

# Response
{
  "result_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAA...",
  "size_recommendation": "M",
  "measurements_used": {
    "height": 170,
    "chest": 90,
    "waist": 75,
    "hips": 95
  }
}
```

#### 4. Measurements
```bash
# Save User Measurements
POST /api/measurements
Authorization: Bearer {token}
Content-Type: application/json

{
  "height": 170.0,
  "weight": 65.0,
  "chest": 90.0,
  "waist": 75.0,
  "hips": 95.0,
  "shoulder_width": 45.0
}

# Extract Measurements from Image
POST /api/extract-measurements
Authorization: Bearer {token}
Content-Type: application/x-www-form-urlencoded

user_image_base64=iVBORw0KGgoAAAANSUhEUgAAAA...
```

---

## Brand Integration Setup

### 1. Brand Registration
Contact VirtualFit support to register your brand and receive:
- Brand API credentials
- Custom brand configuration
- Access to admin dashboard
- Integration documentation

### 2. Product Catalog Integration

#### Option A: API Integration
```javascript
// Sync products from your e-commerce platform
const syncProducts = async () => {
  const products = await fetchFromYourEcommerce();
  
  for (const product of products) {
    await fetch('/api/products', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${brandToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: product.name,
        category: product.category,
        sizes: product.available_sizes,
        image_url: product.main_image,
        description: product.description,
        price: product.price,
        brand_id: 'your_brand_id',
        sku: product.sku
      })
    });
  }
};
```

#### Option B: Webhook Integration
```bash
# Your e-commerce platform sends webhooks on product updates
POST https://virtualfit.com/api/webhooks/products
Authorization: Bearer {webhook_secret}

{
  "event": "product.created",
  "product": {
    "id": "prod_123",
    "name": "Blue Jeans",
    "category": "pants",
    "sizes": ["28", "30", "32", "34"],
    "image_url": "https://brand.com/jeans.jpg"
  }
}
```

---

## Frontend Widget Integration

### 1. JavaScript Widget (Recommended)

```html
<!-- Include VirtualFit Widget -->
<script src="https://virtual-tryon-app.preview.emergentagent.com/widget.js"></script>

<!-- Add Try-On Button to Product Pages -->
<button id="virtualfit-tryon" data-product-id="product_123">
  Try On Virtually
</button>

<script>
VirtualFit.init({
  brandId: 'your_brand_id',
  apiKey: 'your_api_key',
  mode: 'overlay', // 'overlay', 'popup', 'inline'
  customStyles: {
    primaryColor: '#your_brand_color',
    fontFamily: 'your_brand_font'
  }
});

// Initialize try-on button
VirtualFit.attachToButton('#virtualfit-tryon', {
  onSuccess: (result) => {
    console.log('Try-on completed:', result);
    // Show result, add to cart, etc.
  },
  onError: (error) => {
    console.error('Try-on failed:', error);
  }
});
</script>
```

### 2. React Component Integration

```jsx
import { VirtualFitWidget } from '@virtualfit/react-widget';

const ProductPage = ({ product }) => {
  const handleTryOnComplete = (result) => {
    // Handle try-on result
    console.log('Size recommendation:', result.size_recommendation);
    // Update product size selection
    // Show confidence score
    // Add to cart with recommended size
  };

  return (
    <div className="product-page">
      <div className="product-images">
        <img src={product.image} alt={product.name} />
      </div>
      
      <div className="product-details">
        <h1>{product.name}</h1>
        <p>{product.description}</p>
        
        <VirtualFitWidget
          productId={product.id}
          brandId="your_brand_id"
          apiKey="your_api_key"
          onComplete={handleTryOnComplete}
          customStyles={{
            primaryColor: '#your_brand_color',
            borderRadius: '8px'
          }}
        />
        
        <div className="size-selection">
          {/* Size options with AI recommendations */}
        </div>
        
        <button className="add-to-cart">Add to Cart</button>
      </div>
    </div>
  );
};
```

### 3. Mobile App Integration (React Native)

```jsx
import { VirtualFitSDK } from '@virtualfit/react-native';

const ProductScreen = ({ product }) => {
  const openVirtualTryOn = async () => {
    try {
      const result = await VirtualFitSDK.startTryOn({
        productId: product.id,
        brandId: 'your_brand_id',
        apiKey: 'your_api_key'
      });
      
      // Handle result
      console.log('Recommended size:', result.size_recommendation);
    } catch (error) {
      console.error('Try-on error:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Image source={{ uri: product.image }} style={styles.productImage} />
      <Text style={styles.productName}>{product.name}</Text>
      
      <TouchableOpacity 
        style={styles.tryOnButton}
        onPress={openVirtualTryOn}
      >
        <Text style={styles.buttonText}>Try On Virtually</Text>
      </TouchableOpacity>
    </View>
  );
};
```

---

## Authentication & Security

### 1. API Key Management
```javascript
// Secure API key storage (server-side)
const VIRTUALFIT_API_KEY = process.env.VIRTUALFIT_API_KEY;
const VIRTUALFIT_BRAND_ID = process.env.VIRTUALFIT_BRAND_ID;

// Generate user tokens server-side
app.post('/api/virtualfit/auth', async (req, res) => {
  const userToken = await generateVirtualFitToken(req.user.id);
  res.json({ token: userToken });
});
```

### 2. Webhook Security
```javascript
// Verify webhook signatures
const crypto = require('crypto');

const verifyWebhook = (payload, signature, secret) => {
  const expectedSignature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );
};
```

---

## Customization Options

### 1. Brand Styling
```css
/* Custom CSS for white-label integration */
.virtualfit-widget {
  --primary-color: #your-brand-color;
  --secondary-color: #your-secondary-color;
  --font-family: 'Your Brand Font', sans-serif;
  --border-radius: 8px;
  --box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.virtualfit-button {
  background: linear-gradient(45deg, #brand-color-1, #brand-color-2);
  border: none;
  padding: 12px 24px;
  border-radius: var(--border-radius);
  font-family: var(--font-family);
  font-weight: 600;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.virtualfit-modal {
  --modal-width: 800px;
  --modal-height: 600px;
  border-radius: var(--border-radius);
  box-shadow: var(--box-shadow);
}
```

### 2. Custom Prompts & Messaging
```javascript
VirtualFit.configure({
  messages: {
    welcome: "Welcome to [Brand Name] Virtual Try-On",
    uploadPrompt: "Upload your photo to see how this looks on you",
    processing: "Creating your personalized try-on...",
    sizeRecommendation: "Based on your measurements, we recommend size {size}",
    tryAnother: "Try another style"
  },
  
  features: {
    allowCamera: true,
    allowUpload: true,
    showMeasurements: true,
    showSizeChart: true,
    showConfidenceScore: true
  }
});
```

---

## Webhooks & Callbacks

### 1. Event Types
```javascript
// Webhook events you can subscribe to
const webhookEvents = [
  'tryon.completed',        // Try-on session completed
  'tryon.failed',           // Try-on session failed
  'user.measurements.updated', // User updated measurements
  'product.viewed',         // Product viewed in try-on
  'size.recommended'        // Size recommendation generated
];
```

### 2. Webhook Handler Example
```javascript
app.post('/webhooks/virtualfit', (req, res) => {
  const event = req.body;
  
  switch (event.type) {
    case 'tryon.completed':
      // Track conversion analytics
      analytics.track('Virtual Try-On Completed', {
        userId: event.data.user_id,
        productId: event.data.product_id,
        sizeRecommendation: event.data.size_recommendation,
        timestamp: event.data.created_at
      });
      break;
      
    case 'size.recommended':
      // Update product recommendations
      await updateSizeRecommendation(
        event.data.user_id,
        event.data.product_id,
        event.data.recommended_size
      );
      break;
  }
  
  res.status(200).send('OK');
});
```

---

## Error Handling

### 1. Common Error Codes
```javascript
const ERROR_CODES = {
  INVALID_IMAGE: 'IMAGE_001',
  PROCESSING_FAILED: 'PROCESS_001',
  PRODUCT_NOT_FOUND: 'PRODUCT_001',
  QUOTA_EXCEEDED: 'QUOTA_001',
  AUTHENTICATION_FAILED: 'AUTH_001'
};

// Error handling in your integration
const handleVirtualFitError = (error) => {
  switch (error.code) {
    case ERROR_CODES.INVALID_IMAGE:
      showMessage('Please upload a clear, full-body photo');
      break;
    case ERROR_CODES.PROCESSING_FAILED:
      showMessage('Try-on failed. Please try again.');
      break;
    case ERROR_CODES.QUOTA_EXCEEDED:
      showMessage('Daily try-on limit reached. Please try again tomorrow.');
      break;
    default:
      showMessage('Something went wrong. Please contact support.');
  }
};
```

### 2. Retry Logic
```javascript
const tryOnWithRetry = async (params, maxRetries = 3) => {
  let lastError;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      const result = await VirtualFit.tryOn(params);
      return result;
    } catch (error) {
      lastError = error;
      
      if (error.code === 'PROCESSING_FAILED' && i < maxRetries - 1) {
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        continue;
      }
      
      throw error;
    }
  }
  
  throw lastError;
};
```

---

## Performance & Scaling

### 1. Caching Strategy
```javascript
// Cache user measurements and preferences
const cacheUserData = {
  measurements: {
    ttl: 30 * 24 * 60 * 60, // 30 days
    key: (userId) => `user:${userId}:measurements`
  },
  
  tryonResults: {
    ttl: 7 * 24 * 60 * 60, // 7 days
    key: (userId, productId) => `tryon:${userId}:${productId}`
  }
};

// Implement caching
const getCachedMeasurements = async (userId) => {
  const cached = await redis.get(cacheUserData.measurements.key(userId));
  return cached ? JSON.parse(cached) : null;
};
```

### 2. Image Optimization
```javascript
// Optimize images before sending to API
const optimizeImage = (imageFile) => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Resize to optimal dimensions (max 1024x1024)
      const maxSize = 1024;
      let { width, height } = img;
      
      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height);
        width *= ratio;
        height *= ratio;
      }
      
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob(resolve, 'image/jpeg', 0.8);
    };
    
    img.src = URL.createObjectURL(imageFile);
  });
};
```

---

## Testing & Implementation

### 1. Sandbox Environment
```javascript
// Use sandbox for testing
VirtualFit.init({
  environment: 'sandbox', // 'sandbox' or 'production'
  brandId: 'test_brand_123',
  apiKey: 'test_key_456'
});
```

### 2. A/B Testing
```javascript
// Implement A/B testing for virtual try-on placement
const runABTest = () => {
  const variant = Math.random() < 0.5 ? 'A' : 'B';
  
  if (variant === 'A') {
    // Show try-on button prominently
    showTryOnButton({ position: 'top', size: 'large' });
  } else {
    // Show try-on button subtly
    showTryOnButton({ position: 'bottom', size: 'medium' });
  }
  
  // Track variant performance
  analytics.track('VirtualFit AB Test', { variant });
};
```

### 3. Analytics Integration
```javascript
// Track key metrics
const trackVirtualFitEvents = {
  buttonClicked: (productId) => {
    analytics.track('Virtual Try-On Button Clicked', {
      productId,
      timestamp: Date.now()
    });
  },
  
  tryonCompleted: (result) => {
    analytics.track('Virtual Try-On Completed', {
      productId: result.productId,
      sizeRecommendation: result.size_recommendation,
      userSatisfaction: result.confidence_score,
      conversionLikelihood: result.conversion_score
    });
  },
  
  addToCartAfterTryon: (productId, size) => {
    analytics.track('Add to Cart After Virtual Try-On', {
      productId,
      recommendedSize: size,
      conversionSource: 'virtual_tryon'
    });
  }
};
```

---

## Support & Maintenance

### 1. Monitoring & Alerts
```javascript
// Set up monitoring for your integration
const monitorVirtualFit = {
  checkHealth: async () => {
    try {
      const response = await fetch('/api/', {
        headers: { 'Authorization': `Bearer ${apiKey}` }
      });
      return response.ok;
    } catch (error) {
      console.error('VirtualFit health check failed:', error);
      return false;
    }
  },
  
  trackUsage: (event) => {
    // Monitor API usage and performance
    monitoring.record('virtualfit.api.usage', {
      endpoint: event.endpoint,
      responseTime: event.duration,
      statusCode: event.statusCode
    });
  }
};
```

### 2. Support Contacts
- **Technical Support**: support@virtualfit.com
- **Integration Help**: integrations@virtualfit.com
- **API Documentation**: https://docs.virtualfit.com
- **Status Page**: https://status.virtualfit.com

### 3. SLA & Service Levels
- **API Uptime**: 99.9% guaranteed
- **Response Time**: < 3 seconds for try-on processing
- **Support Response**: < 24 hours for integration issues
- **Rate Limits**: 1000 requests/hour per brand (upgradeable)

---

## Quick Start Checklist

### For Brand Integration:
- [ ] Register brand account and get API credentials
- [ ] Set up product catalog sync
- [ ] Implement frontend widget on product pages
- [ ] Configure webhooks for analytics
- [ ] Test in sandbox environment
- [ ] Deploy to production
- [ ] Monitor performance and user adoption

### For E-commerce Platform Integration:
- [ ] Install VirtualFit plugin/extension
- [ ] Configure brand settings
- [ ] Map product categories and sizes
- [ ] Customize styling to match brand
- [ ] Set up analytics tracking
- [ ] Train customer support team
- [ ] Launch with marketing campaign

---

## Conclusion

VirtualFit provides a comprehensive virtual try-on solution that can be seamlessly integrated into any e-commerce platform. With robust APIs, customizable widgets, and enterprise-grade security, brands can offer their customers an engaging and accurate virtual try-on experience that drives conversions and reduces returns.

For implementation support or custom integration requirements, contact our team at integrations@virtualfit.com.

---

*Last Updated: January 2025*
*Version: 2.0*