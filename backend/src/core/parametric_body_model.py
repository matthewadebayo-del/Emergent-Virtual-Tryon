"""
Parametric Body Models (100% Open Source)
Mathematical approach using only open source libraries - Zero licensing issues!
"""

import numpy as np
import trimesh  # MIT License ✅
from scipy.interpolate import splprep, splev  # BSD License ✅

class ParametricHumanModel:
    """
    Create human body models using mathematical functions
    Completely original - no licensing restrictions!
    """
    
    def __init__(self):
        # Standard human proportions (from art/anatomy books - public domain)
        self.proportions = {
            'head_height': 1.0,      # 1 head unit
            'torso_height': 3.0,     # 3 head units  
            'leg_height': 4.0,       # 4 head units
            'shoulder_width': 2.0,   # 2 head units
            'hip_width': 1.5,        # 1.5 head units
        }
    
    def create_parametric_body(self, height_cm=170, measurements=None):
        """Create body using parametric equations"""
        # Calculate head unit size
        head_unit = height_cm / 8.0  # Standard 8-head-tall figure
        
        # Generate body sections
        head = self.generate_head(head_unit)
        torso = self.generate_torso(head_unit, measurements)
        arms = self.generate_arms(head_unit, measurements)
        legs = self.generate_legs(head_unit, measurements)
        
        # Combine all parts
        body_parts = [head, torso] + arms + legs
        full_body = trimesh.util.concatenate(body_parts)
        
        return full_body
    
    def generate_head(self, head_unit):
        """Generate head using ellipsoid"""
        head = trimesh.creation.icosphere(radius=head_unit * 0.5, subdivisions=2)
        head.apply_scale([0.85, 1.0, 1.1])  # width, depth, height
        head.apply_translation([0, 0, head_unit * 7.5])
        return head
    
    def generate_torso(self, head_unit, measurements=None):
        """Generate torso using parametric curves"""
        
        if measurements:
            chest_width = measurements.get('shoulder_width', head_unit * 2) / 2
            waist_width = measurements.get('waist_circumference', head_unit * 1.5) / (2 * np.pi)
            hip_width = measurements.get('hip_circumference', head_unit * 1.8) / (2 * np.pi)
        else:
            chest_width = head_unit * 1.0
            waist_width = head_unit * 0.75
            hip_width = head_unit * 0.9
        
        # Create torso profile curve
        heights = np.array([6.5, 5.5, 4.5, 3.5]) * head_unit
        widths = np.array([chest_width, chest_width * 0.9, waist_width, hip_width])
        
        # Generate smooth torso using revolution
        torso = self.create_revolution_surface(heights, widths)
        return torso
    
    def create_revolution_surface(self, heights, radii):
        """Create 3D surface by revolving a profile curve"""
        profile_points = list(zip(radii, heights))
        theta = np.linspace(0, 2*np.pi, 32)
        vertices = []
        faces = []
        
        for i, (r, h) in enumerate(profile_points):
            for j, angle in enumerate(theta):
                x = r * np.cos(angle)
                y = r * np.sin(angle) 
                z = h
                vertices.append([x, y, z])
        
        # Generate faces
        n_theta = len(theta)
        for i in range(len(profile_points) - 1):
            for j in range(n_theta):
                v1 = i * n_theta + j
                v2 = i * n_theta + (j + 1) % n_theta
                v3 = (i + 1) * n_theta + (j + 1) % n_theta
                v4 = (i + 1) * n_theta + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def generate_arms(self, head_unit, measurements=None):
        """Generate arms using cylindrical approximation"""
        
        if measurements:
            arm_length = measurements.get('arm_length', head_unit * 2.5)
        else:
            arm_length = head_unit * 2.5
        
        upper_arm_length = arm_length * 0.6
        forearm_length = arm_length * 0.4
        
        # Left arm
        left_upper = trimesh.creation.cylinder(radius=head_unit * 0.15, height=upper_arm_length)
        left_upper.apply_rotation([np.pi/2, 0, 0])
        left_upper.apply_translation([-head_unit * 1.2, 0, head_unit * 5.5])
        
        left_forearm = trimesh.creation.cylinder(radius=head_unit * 0.12, height=forearm_length)
        left_forearm.apply_rotation([np.pi/2, 0, 0])
        left_forearm.apply_translation([-head_unit * 1.8, 0, head_unit * 5.5])
        
        # Right arm (mirror)
        right_upper = left_upper.copy()
        right_upper.apply_translation([head_unit * 2.4, 0, 0])
        
        right_forearm = left_forearm.copy()
        right_forearm.apply_translation([head_unit * 3.6, 0, 0])
        
        return [left_upper, left_forearm, right_upper, right_forearm]
    
    def generate_legs(self, head_unit, measurements=None):
        """Generate legs using tapered cylinders"""
        
        leg_length = head_unit * 4.0
        thigh_length = leg_length * 0.55
        shin_length = leg_length * 0.45
        
        # Left leg
        left_thigh = trimesh.creation.cylinder(radius=head_unit * 0.18, height=thigh_length)
        left_thigh.apply_translation([-head_unit * 0.25, 0, head_unit * 2.0])
        
        left_shin = trimesh.creation.cylinder(radius=head_unit * 0.14, height=shin_length)
        left_shin.apply_translation([-head_unit * 0.25, 0, head_unit * 0.7])
        
        # Right leg (mirror)
        right_thigh = left_thigh.copy()
        right_thigh.apply_translation([head_unit * 0.5, 0, 0])
        
        right_shin = left_shin.copy() 
        right_shin.apply_translation([head_unit * 0.5, 0, 0])
        
        return [left_thigh, left_shin, right_thigh, right_shin]


class EllipsoidBodyModel:
    """
    Super simple but effective body model using only ellipsoids
    Perfect for garment try-on - no complex licensing
    """
    
    def create_simple_body(self, measurements):
        """Create body using simple ellipsoids"""
        
        # Extract key measurements
        height = measurements.get('height', 170)
        chest = measurements.get('chest_circumference', 90) 
        waist = measurements.get('waist_circumference', 75)
        hips = measurements.get('hip_circumference', 95)
        
        # Scale everything relative to height
        scale = height / 170.0
        
        # Create body parts as ellipsoids
        parts = []
        
        # Head
        head = trimesh.creation.icosphere(radius=9 * scale, subdivisions=1)
        head.apply_scale([0.85, 1.0, 1.1])
        head.apply_translation([0, 0, 160 * scale])
        parts.append(head)
        
        # Torso (chest area)
        chest_radius = chest / (2 * np.pi)
        torso = trimesh.creation.icosphere(radius=chest_radius * scale, subdivisions=2)
        torso.apply_scale([1.0, 0.6, 1.8])
        torso.apply_translation([0, 0, 120 * scale])
        parts.append(torso)
        
        # Waist
        waist_radius = waist / (2 * np.pi)
        waist_part = trimesh.creation.icosphere(radius=waist_radius * scale, subdivisions=2)
        waist_part.apply_scale([1.0, 0.6, 1.2])
        waist_part.apply_translation([0, 0, 95 * scale])
        parts.append(waist_part)
        
        # Hips  
        hip_radius = hips / (2 * np.pi)
        hip_part = trimesh.creation.icosphere(radius=hip_radius * scale, subdivisions=2)
        hip_part.apply_scale([1.0, 0.7, 1.0])
        hip_part.apply_translation([0, 0, 75 * scale])
        parts.append(hip_part)
        
        # Simple arms (cylinders)
        for side in [-1, 1]:  # Left and right
            arm = trimesh.creation.cylinder(radius=6 * scale, height=55 * scale)
            arm.apply_rotation([0, np.pi/2, 0])
            arm.apply_translation([side * 35 * scale, 0, 120 * scale])
            parts.append(arm)
        
        # Simple legs (cylinders)
        for side in [-0.5, 0.5]:  # Left and right
            leg = trimesh.creation.cylinder(radius=8 * scale, height=75 * scale)
            leg.apply_translation([side * 15 * scale, 0, 35 * scale])
            parts.append(leg)
        
        # Combine all parts
        body = trimesh.util.concatenate(parts)
        return body


def create_license_free_body(user_measurements):
    """Create a completely license-free 3D body model"""
    
    # Use simple ellipsoid approach (fastest and most reliable)
    ellipsoid_model = EllipsoidBodyModel()
    body = ellipsoid_model.create_simple_body(user_measurements)
    
    print("✅ License-free body model created!")
    return body


def validate_measurements(measurements):
    """Validate and correct measurements using anatomical knowledge"""
    corrected = measurements.copy()
    
    # Ensure reasonable proportions
    if 'height' in measurements and 'shoulder_width' in measurements:
        expected_shoulder = measurements['height'] * 0.25
        if abs(measurements['shoulder_width'] - expected_shoulder) > 20:
            corrected['shoulder_width'] = expected_shoulder
            print(f"⚠️ Corrected shoulder width to {expected_shoulder:.1f}cm")
    
    return corrected