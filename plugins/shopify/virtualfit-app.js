/**
 * VirtualFit Shopify App
 * Shopify integration for virtual try-on
 */

// App configuration
const VIRTUALFIT_CONFIG = {
  apiKey: window.VIRTUALFIT_API_KEY || '',
  baseUrl: 'https://api.virtualfit.com/api/v1'
};

class ShopifyVirtualFit {
  constructor() {
    this.sdk = new VirtualFitSDK(VIRTUALFIT_CONFIG);
    this.init();
  }

  init() {
    // Wait for Shopify to load
    document.addEventListener('DOMContentLoaded', () => {
      this.addTryOnButtons();
      this.setupEventListeners();
    });
  }

  addTryOnButtons() {
    // Add try-on button to product pages
    const productForms = document.querySelectorAll('form[action*="/cart/add"]');
    
    productForms.forEach(form => {
      const productData = this.extractProductData(form);
      if (this.isTryOnEligible(productData)) {
        this.insertTryOnButton(form, productData);
      }
    });
  }

  extractProductData(form) {
    const productJson = document.querySelector('#product-json');
    if (productJson) {
      const product = JSON.parse(productJson.textContent);
      const selectedVariant = this.getSelectedVariant(product, form);
      
      return {
        id: product.id,
        name: product.title,
        category: this.mapProductType(product.product_type),
        imageUrl: selectedVariant?.featured_image?.src || product.featured_image,
        color: this.extractColor(selectedVariant),
        size: this.extractSize(selectedVariant)
      };
    }
    return null;
  }

  getSelectedVariant(product, form) {
    const variantId = form.querySelector('[name="id"]')?.value;
    return product.variants.find(v => v.id == variantId) || product.variants[0];
  }

  mapProductType(productType) {
    const typeMap = {
      'shirts': 'TOP',
      't-shirts': 'TOP', 
      'tops': 'TOP',
      'pants': 'BOTTOM',
      'jeans': 'BOTTOM',
      'dresses': 'DRESS',
      'shoes': 'SHOES',
      'jackets': 'OUTERWEAR'
    };
    return typeMap[productType?.toLowerCase()] || 'TOP';
  }

  extractColor(variant) {
    if (!variant) return 'default';
    
    const colorOption = variant.options?.find(opt => 
      opt.name?.toLowerCase().includes('color') || 
      opt.name?.toLowerCase().includes('colour')
    );
    return colorOption?.value?.toLowerCase() || 'default';
  }

  extractSize(variant) {
    if (!variant) return 'M';
    
    const sizeOption = variant.options?.find(opt => 
      opt.name?.toLowerCase().includes('size')
    );
    return sizeOption?.value || 'M';
  }

  isTryOnEligible(productData) {
    return productData && ['TOP', 'BOTTOM', 'DRESS', 'OUTERWEAR'].includes(productData.category);
  }

  insertTryOnButton(form, productData) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn virtualfit-btn';
    button.innerHTML = '👕 Try On Virtual';
    button.onclick = () => this.openTryOnModal(productData);
    
    const addToCartBtn = form.querySelector('[type="submit"]');
    if (addToCartBtn) {
      addToCartBtn.parentNode.insertBefore(button, addToCartBtn.nextSibling);
    }
  }

  openTryOnModal(productData) {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'virtualfit-modal';
    modal.innerHTML = `
      <div class="virtualfit-modal-content">
        <span class="virtualfit-close">&times;</span>
        <h3>Virtual Try-On: ${productData.name}</h3>
        <div class="virtualfit-upload-area">
          <input type="file" id="virtualfit-upload" accept="image/*" style="display:none">
          <button onclick="document.getElementById('virtualfit-upload').click()" class="btn">
            📷 Upload Your Photo
          </button>
          <p>Or take a photo with your camera</p>
          <button onclick="this.startCamera()" class="btn">📱 Use Camera</button>
        </div>
        <div id="virtualfit-result" style="display:none">
          <img id="virtualfit-result-img" style="max-width:100%">
          <button onclick="this.addToCart()" class="btn btn-primary">Add to Cart</button>
        </div>
        <div id="virtualfit-loading" style="display:none">Processing your try-on...</div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // Event listeners
    modal.querySelector('.virtualfit-close').onclick = () => {
      document.body.removeChild(modal);
    };
    
    modal.querySelector('#virtualfit-upload').onchange = (e) => {
      this.processImage(e.target.files[0], productData, modal);
    };
  }

  async processImage(file, productData, modal) {
    const loading = modal.querySelector('#virtualfit-loading');
    loading.style.display = 'block';
    
    try {
      // Upload customer image
      const customerImageUrl = await this.uploadToShopify(file);
      
      // Process try-on
      const result = await this.sdk.processSync(
        customerImageUrl,
        productData.imageUrl,
        productData
      );
      
      if (result.success) {
        this.displayResult(result.result_image_base64, modal);
      } else {
        throw new Error(result.error || 'Processing failed');
      }
    } catch (error) {
      alert('Try-on failed: ' + error.message);
    } finally {
      loading.style.display = 'none';
    }
  }

  async uploadToShopify(file) {
    // Use Shopify's file upload API or CDN
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/admin/api/2023-10/files.json', {
      method: 'POST',
      body: formData,
      headers: {
        'X-Shopify-Access-Token': window.SHOPIFY_ACCESS_TOKEN
      }
    });
    
    const data = await response.json();
    return data.file.src;
  }

  displayResult(base64Image, modal) {
    const resultDiv = modal.querySelector('#virtualfit-result');
    const resultImg = modal.querySelector('#virtualfit-result-img');
    
    resultImg.src = base64Image;
    resultDiv.style.display = 'block';
  }

  setupEventListeners() {
    // Listen for variant changes
    document.addEventListener('change', (e) => {
      if (e.target.name === 'id' || e.target.classList.contains('product-variant-option')) {
        // Refresh try-on buttons for new variant
        setTimeout(() => this.addTryOnButtons(), 100);
      }
    });
  }
}

// Initialize when script loads
if (typeof VirtualFitSDK !== 'undefined') {
  new ShopifyVirtualFit();
} else {
  console.error('VirtualFit SDK not loaded');
}