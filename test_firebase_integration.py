#!/usr/bin/env python3
"""
Test Firebase integration and L.L.Bean catalog
"""
import sys
import os
sys.path.append('backend')

def test_firebase_wrapper():
    """Test Firebase wrapper functionality"""
    print("Testing Firebase wrapper...")
    from backend.firebase_db import FirebaseDB
    
    db = FirebaseDB(None)
    print(f"✅ Mock mode initialized: {db.mock_mode}")
    
    users = db.users
    products = db.products
    print("✅ Collection access working")
    
    return True

def test_llbean_catalog():
    """Test L.L.Bean product catalog"""
    print("Testing L.L.Bean catalog...")
    from backend.llbean_catalog import get_llbean_products, get_mens_shirts, get_womens_blouses
    
    products = get_llbean_products()
    print(f"✅ Found {len(products)} total products")
    
    mens_shirts = get_mens_shirts()
    print(f"✅ Found {len(mens_shirts)} men's shirts")
    
    womens_blouses = get_womens_blouses()
    print(f"✅ Found {len(womens_blouses)} women's blouses")
    
    for product in products[:2]:
        print(f"  - {product['name']} ({product['category']}) - ${product['price']}")
    
    return len(products) == 8

def main():
    """Run all tests"""
    print("🧪 Testing Firebase Integration and L.L.Bean Catalog")
    print("=" * 60)
    
    try:
        firebase_ok = test_firebase_wrapper()
        print()
        
        catalog_ok = test_llbean_catalog()
        print()
        
        if firebase_ok and catalog_ok:
            print("✅ All tests passed! Firebase integration and L.L.Bean catalog ready.")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
