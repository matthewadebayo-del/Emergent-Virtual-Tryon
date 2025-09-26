/**
 * VirtualFit JavaScript SDK
 * E-commerce integration for virtual try-on
 */
class VirtualFitSDK {
  constructor(config) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.virtualfit.com/api/v1';
    this.webhookUrl = config.webhookUrl;
  }

  async processAsync(customerImageUrl, garmentImageUrl, productInfo) {
    const response = await fetch(`${this.baseUrl}/tryon/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: this.apiKey,
        customer_image_url: customerImageUrl,
        garment_image_url: garmentImageUrl,
        product_info: productInfo,
        webhook_url: this.webhookUrl
      })
    });
    return response.json();
  }

  async processSync(customerImageUrl, garmentImageUrl, productInfo) {
    const response = await fetch(`${this.baseUrl}/tryon/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: this.apiKey,
        customer_image_url: customerImageUrl,
        garment_image_url: garmentImageUrl,
        product_info: productInfo
      })
    });
    return response.json();
  }

  async getStatus(jobId) {
    const response = await fetch(`${this.baseUrl}/tryon/status/${jobId}`);
    return response.json();
  }

  async getResult(jobId) {
    const response = await fetch(`${this.baseUrl}/tryon/result/${jobId}`);
    return response.json();
  }

  // Widget integration
  createTryOnWidget(containerId, productData) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
      <div class="virtualfit-widget">
        <button id="vf-try-on-btn" class="vf-btn">Try On Virtual</button>
        <div id="vf-modal" class="vf-modal" style="display:none">
          <div class="vf-modal-content">
            <span class="vf-close">&times;</span>
            <input type="file" id="vf-image-upload" accept="image/*">
            <canvas id="vf-result-canvas"></canvas>
            <div id="vf-loading">Processing...</div>
          </div>
        </div>
      </div>
    `;

    document.getElementById('vf-try-on-btn').onclick = () => {
      document.getElementById('vf-modal').style.display = 'block';
    };

    document.querySelector('.vf-close').onclick = () => {
      document.getElementById('vf-modal').style.display = 'none';
    };

    document.getElementById('vf-image-upload').onchange = async (e) => {
      const file = e.target.files[0];
      if (file) {
        const formData = new FormData();
        formData.append('image', file);
        
        // Upload customer image and process
        const customerImageUrl = await this.uploadImage(formData);
        const result = await this.processSync(customerImageUrl, productData.imageUrl, productData);
        
        if (result.success) {
          this.displayResult(result.result_image_base64);
        }
      }
    };
  }

  async uploadImage(formData) {
    // Implement image upload to your CDN
    return 'https://cdn.example.com/uploaded-image.jpg';
  }

  displayResult(base64Image) {
    const canvas = document.getElementById('vf-result-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = base64Image;
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = VirtualFitSDK;
} else if (typeof window !== 'undefined') {
  window.VirtualFitSDK = VirtualFitSDK;
}