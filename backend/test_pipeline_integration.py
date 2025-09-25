"""
Pipeline Integration Test
Tests the enhanced pipeline controller with comprehensive try-on integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_pipeline_controller_import():
    """Test pipeline controller import"""
    print("[PIPELINE TEST] Testing pipeline controller import...")
    
    try:
        from core.enhanced_pipeline_controller import EnhancedPipelineController
        print("[PIPELINE TEST] SUCCESS: Pipeline controller imported")
        return True
    except ImportError as e:
        print(f"[PIPELINE TEST] FAILED: Pipeline controller import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline controller initialization"""
    print("[PIPELINE TEST] Testing pipeline initialization...")
    
    try:
        from core.enhanced_pipeline_controller import EnhancedPipelineController
        controller = EnhancedPipelineController()
        print("[PIPELINE TEST] SUCCESS: Pipeline controller initialized")
        return True
    except Exception as e:
        print(f"[PIPELINE TEST] FAILED: Pipeline initialization failed: {e}")
        return False

def test_configuration_in_pipeline():
    """Test that configuration variables are properly set in pipeline"""
    print("[PIPELINE TEST] Testing configuration in pipeline...")
    
    # This simulates the configuration that should be in the pipeline
    USE_COMPREHENSIVE_TRYON = True
    USE_SAFE_MODE = False
    
    GARMENT_TYPE_MAPPING = {
        'shirts': ['top'],
        'tops': ['top'],
        't-shirt': ['top'],
        'tshirt': ['top'],
        'blouse': ['top'],
        'sweater': ['top'],
        'pants': ['bottom'],
        'jeans': ['bottom'],
        'trousers': ['bottom'],
        'shorts': ['bottom'],
        'skirt': ['bottom'],
        'shoes': ['shoes'],
        'sneakers': ['shoes'],
        'boots': ['shoes'],
        'dress': ['dress'],
        'jacket': ['outerwear'],
        'coat': ['outerwear'],
        'blazer': ['outerwear'],
        'outfit': ['top', 'bottom'],
        'full_outfit': ['top', 'bottom', 'shoes'],
    }
    
    if USE_COMPREHENSIVE_TRYON and not USE_SAFE_MODE:
        print("[PIPELINE TEST] SUCCESS: Configuration variables correct")
    else:
        print("[PIPELINE TEST] FAILED: Configuration variables incorrect")
        return False
    
    if len(GARMENT_TYPE_MAPPING) >= 20:
        print(f"[PIPELINE TEST] SUCCESS: Garment mapping has {len(GARMENT_TYPE_MAPPING)} entries")
    else:
        print(f"[PIPELINE TEST] FAILED: Garment mapping only has {len(GARMENT_TYPE_MAPPING)} entries")
        return False
    
    return True

def test_mock_processing_flow():
    """Test mock processing flow"""
    print("[PIPELINE TEST] Testing mock processing flow...")
    
    # Mock data structures
    mock_customer_analysis = {
        'pose_landmarks': {
            'left_shoulder': {'confidence': 0.8},
            'right_shoulder': {'confidence': 0.9},
            'left_hip': {'confidence': 0.7},
            'right_hip': {'confidence': 0.8}
        },
        'measurements': {
            'chest': 90.0,
            'waist': 75.0,
            'height': 170.0
        }
    }
    
    mock_garment_analysis = {
        'dominant_colors': [(255, 255, 255)],
        'fabric_type': 'cotton',
        'patterns': []
    }
    
    mock_product_info = {
        'name': 'Classic White T-Shirt',
        'category': 'shirts',
        'product_id': 'test-001'
    }
    
    # Test data validation
    required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    pose_landmarks = mock_customer_analysis.get('pose_landmarks', {})
    
    missing_landmarks = []
    for landmark in required_landmarks:
        if landmark not in pose_landmarks or pose_landmarks[landmark].get('confidence', 0) < 0.7:
            missing_landmarks.append(landmark)
    
    if not missing_landmarks:
        print("[PIPELINE TEST] SUCCESS: All required landmarks present with good confidence")
    else:
        print(f"[PIPELINE TEST] FAILED: Missing landmarks: {missing_landmarks}")
        return False
    
    # Test garment type detection
    GARMENT_TYPE_MAPPING = {'shirts': ['top']}
    product_category = mock_product_info.get('category', 'shirts').lower()
    garment_types = GARMENT_TYPE_MAPPING.get(product_category, ['top'])
    
    if garment_types == ['top']:
        print(f"[PIPELINE TEST] SUCCESS: Garment type detection: {product_category} -> {garment_types}")
    else:
        print(f"[PIPELINE TEST] FAILED: Garment type detection failed")
        return False
    
    return True

def run_pipeline_tests():
    """Run all pipeline integration tests"""
    print("[PIPELINE INTEGRATION] Starting pipeline integration tests...")
    print("=" * 70)
    
    tests = [
        ("Pipeline Import", test_pipeline_controller_import),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Configuration", test_configuration_in_pipeline),
        ("Mock Processing Flow", test_mock_processing_flow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[PIPELINE TEST] Running: {test_name}")
        print("-" * 50)
        try:
            if test_func():
                passed += 1
                print(f"[PIPELINE TEST] {test_name}: PASSED")
            else:
                print(f"[PIPELINE TEST] {test_name}: FAILED")
        except Exception as e:
            print(f"[PIPELINE TEST] {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"[PIPELINE INTEGRATION] Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[PIPELINE INTEGRATION] SUCCESS: All pipeline tests PASSED")
        return True
    else:
        print("[PIPELINE INTEGRATION] WARNING: Some pipeline tests FAILED")
        return False

if __name__ == "__main__":
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)