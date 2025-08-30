"""
L.L.Bean Product Catalog
Real product data for testing virtual try-on functionality
"""

LLBEAN_PRODUCTS = [
    {
        "id": "llbean_mens_signature_polo_navy",
        "name": "Men's Signature Polo Shirt - Navy",
        "category": "shirt",
        "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
        "image_url": "https://global.llbean.com/llb/shop/516693?page=mens-signature-polo-shirt&bc=12-26&feat=26-GN1&csp=f",
        "description": "Classic fit polo shirt made from premium cotton pique. Features three-button placket and ribbed collar.",
        "price": 39.95,
        "color": "Navy",
        "material": "100% Cotton",
        "fit": "Classic Fit"
    },
    {
        "id": "llbean_mens_katahdin_flannel_red",
        "name": "Men's Katahdin Iron Works Flannel Shirt - Red Plaid",
        "category": "shirt", 
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "image_url": "https://global.llbean.com/llb/shop/42272?page=mens-katahdin-iron-works-flannel-shirt&bc=12-26&feat=26-GN1",
        "description": "Heavyweight flannel shirt with classic plaid pattern. Button-down collar and chest pocket.",
        "price": 59.95,
        "color": "Red Plaid",
        "material": "100% Cotton Flannel",
        "fit": "Traditional Fit"
    },
    {
        "id": "llbean_mens_chamois_shirt_blue",
        "name": "Men's Chamois Shirt - Blue",
        "category": "shirt",
        "sizes": ["S", "M", "L", "XL", "XXL", "XXXL"],
        "image_url": "https://global.llbean.com/llb/shop/505?page=mens-chamois-shirt&bc=12-26&feat=26-GN1",
        "description": "Soft, brushed cotton chamois shirt. Classic button-front design with chest pocket.",
        "price": 49.95,
        "color": "Blue",
        "material": "100% Cotton Chamois",
        "fit": "Traditional Fit"
    },
    {
        "id": "llbean_mens_oxford_shirt_white",
        "name": "Men's Wrinkle-Free Pinpoint Oxford Shirt - White",
        "category": "shirt",
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "image_url": "https://global.llbean.com/llb/shop/516694?page=mens-wrinkle-free-pinpoint-oxford-shirt&bc=12-26",
        "description": "Crisp oxford shirt with wrinkle-free finish. Button-down collar and barrel cuffs.",
        "price": 54.95,
        "color": "White",
        "material": "100% Cotton Oxford",
        "fit": "Traditional Fit"
    },
    
    {
        "id": "llbean_womens_signature_blouse_navy",
        "name": "Women's Signature Cotton/Modal Blouse - Navy",
        "category": "blouse",
        "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
        "image_url": "https://global.llbean.com/llb/shop/516695?page=womens-signature-cotton-modal-blouse&bc=12-27",
        "description": "Elegant blouse made from cotton-modal blend. Features button-front closure and curved hem.",
        "price": 49.95,
        "color": "Navy",
        "material": "60% Cotton, 40% Modal",
        "fit": "Classic Fit"
    },
    {
        "id": "llbean_womens_wrinkle_free_shirt_white",
        "name": "Women's Wrinkle-Free Pinpoint Oxford Shirt - White",
        "category": "blouse",
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image_url": "https://global.llbean.com/llb/shop/516696?page=womens-wrinkle-free-pinpoint-oxford-shirt&bc=12-27",
        "description": "Professional oxford shirt with wrinkle-free technology. Button-down collar and tailored fit.",
        "price": 52.95,
        "color": "White",
        "material": "100% Cotton Oxford",
        "fit": "Tailored Fit"
    },
    {
        "id": "llbean_womens_chamois_shirt_pink",
        "name": "Women's Chamois Shirt - Pink",
        "category": "blouse",
        "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
        "image_url": "https://global.llbean.com/llb/shop/516697?page=womens-chamois-shirt&bc=12-27",
        "description": "Soft brushed cotton chamois shirt. Relaxed fit with button-front closure.",
        "price": 47.95,
        "color": "Pink",
        "material": "100% Cotton Chamois",
        "fit": "Relaxed Fit"
    },
    {
        "id": "llbean_womens_flannel_blouse_plaid",
        "name": "Women's Scotch Plaid Flannel Shirt - Blue Plaid",
        "category": "blouse",
        "sizes": ["XS", "S", "M", "L", "XL"],
        "image_url": "https://global.llbean.com/llb/shop/516698?page=womens-scotch-plaid-flannel-shirt&bc=12-27",
        "description": "Classic flannel shirt in traditional plaid pattern. Button-front with chest pocket.",
        "price": 54.95,
        "color": "Blue Plaid",
        "material": "100% Cotton Flannel",
        "fit": "Classic Fit"
    }
]

def get_llbean_products():
    """Return the complete L.L.Bean product catalog"""
    return LLBEAN_PRODUCTS

def get_mens_shirts():
    """Return only men's shirts from the catalog"""
    return [product for product in LLBEAN_PRODUCTS if product["category"] == "shirt"]

def get_womens_blouses():
    """Return only women's blouses from the catalog"""
    return [product for product in LLBEAN_PRODUCTS if product["category"] == "blouse"]

def get_product_by_id(product_id: str):
    """Get a specific product by ID"""
    for product in LLBEAN_PRODUCTS:
        if product["id"] == product_id:
            return product
    return None
