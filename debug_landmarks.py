#!/usr/bin/env python3
"""
Debug script to check landmark format in customer analysis
"""

def debug_customer_analysis():
    """Debug what the customer analysis actually contains"""
    
    # Simulate the customer analysis data from the logs
    customer_analysis = {
        'pose_landmarks': {
            # This should contain the actual landmarks
        },
        'pose_keypoints': {
            'left_shoulder': {'x': 0.3, 'y': 0.2, 'confidence': 0.95},
            'right_shoulder': {'x': 0.7, 'y': 0.2, 'confidence': 0.95},
            'left_hip': {'x': 0.35, 'y': 0.6, 'confidence': 0.95},
            'right_hip': {'x': 0.65, 'y': 0.6, 'confidence': 0.95},
        },
        'measurements': {
            'shoulder_width_cm': 45.0,
            'height_cm': 167.75,
        }
    }
    
    print("=== DEBUGGING LANDMARK FORMAT ===")
    print(f"Customer analysis keys: {list(customer_analysis.keys())}")
    print(f"pose_landmarks: {customer_analysis.get('pose_landmarks', 'MISSING')}")
    print(f"pose_keypoints: {customer_analysis.get('pose_keypoints', 'MISSING')}")
    
    # Test the fallback logic
    pose_landmarks = customer_analysis.get('pose_landmarks', customer_analysis.get('pose_keypoints', {}))
    print(f"Final pose_landmarks: {pose_landmarks}")
    
    # Test landmark validation
    required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    missing_landmarks = []
    
    for landmark in required_landmarks:
        if landmark not in pose_landmarks:
            missing_landmarks.append(landmark)
        else:
            confidence = pose_landmarks[landmark].get('confidence', 0)
            print(f"{landmark}: confidence = {confidence}")
            if confidence < 0.7:
                missing_landmarks.append(landmark)
    
    print(f"Missing landmarks: {missing_landmarks}")
    
    if missing_landmarks:
        print("❌ WOULD FAIL - Missing critical landmarks")
    else:
        print("✅ WOULD PASS - All landmarks present")

if __name__ == "__main__":
    debug_customer_analysis()