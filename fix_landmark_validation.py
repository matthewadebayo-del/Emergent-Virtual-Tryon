#!/usr/bin/env python3
"""
Fix for landmark validation - add debug logging to see actual data format
"""

def debug_pose_data(customer_analysis_data):
    """Debug the actual pose data format"""
    print(f"[DEBUG] Customer analysis keys: {list(customer_analysis_data.keys())}")
    
    pose_landmarks = customer_analysis_data.get('pose_landmarks', {})
    pose_keypoints = customer_analysis_data.get('pose_keypoints', {})
    
    print(f"[DEBUG] pose_landmarks type: {type(pose_landmarks)}, content: {pose_landmarks}")
    print(f"[DEBUG] pose_keypoints type: {type(pose_keypoints)}, content: {pose_keypoints}")
    
    # Try both formats
    landmarks = pose_landmarks if pose_landmarks else pose_keypoints
    print(f"[DEBUG] Using landmarks: {landmarks}")
    
    if isinstance(landmarks, dict):
        for key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if key in landmarks:
                print(f"[DEBUG] {key}: {landmarks[key]}")
            else:
                print(f"[DEBUG] {key}: MISSING")
    
    return landmarks

# The fix is to add better debugging and handle the actual data format
print("Add this debug code to the production server to see the actual landmark format")