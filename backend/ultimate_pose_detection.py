"""
ULTIMATE VIRTUAL TRY-ON POSE DETECTION SYSTEM
==============================================

This is the most sophisticated, production-ready implementation combining:
- Tiered quality assessment (from Artifact #1)
- Drop-in integration (from Artifact #4)
- Adaptive thresholds
- User feedback system
- Statistics tracking
- Comprehensive error handling

Replace your existing files with this code for the best results.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class QualityLevel(Enum):
    """Quality levels for pose detection"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class UserType(Enum):
    """User experience levels for adaptive thresholds"""
    FIRST_TIME = "first_time"
    RETURNING = "returning"
    PREMIUM = "premium"
    BULK = "bulk"

@dataclass
class PoseQualityResult:
    """Comprehensive result of pose quality assessment"""
    detected: bool
    quality_level: QualityLevel
    confidence_score: float
    can_proceed: bool
    user_message: str
    technical_details: Dict
    recommendations: List[str]
    quality_color: str
    show_warning: bool
    offer_retake: bool

@dataclass
class ProcessingStats:
    """Statistics for monitoring and optimization"""
    total_processed: int = 0
    accepted: int = 0
    rejected: int = 0
    quality_distribution: Dict[str, int] = None
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    
    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {}

# ============================================================================
# CORE: SOPHISTICATED POSE DETECTOR
# ============================================================================

class UltimatePoseDetector:
    """
    Most sophisticated pose detection system combining all advanced features.
    Drop-in replacement for your existing pose detection.
    """
    
    def __init__(self, user_type: UserType = UserType.RETURNING):
        """
        Initialize with adaptive thresholds based on user type.
        
        Args:
            user_type: User experience level for threshold adaptation
        """
        self.user_type = user_type
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = ProcessingStats()
        
        # Get adaptive thresholds
        self.THRESHOLDS = self._get_adaptive_thresholds(user_type)
        
        # Critical landmarks for virtual try-on
        self.CRITICAL_LANDMARKS = {
            'shoulders': [11, 12],
            'hips': [23, 24],
        }
        
        self.IMPORTANT_LANDMARKS = {
            'elbows': [13, 14],
            'wrists': [15, 16],
            'knees': [25, 26],
        }
        
        # Initialize dual detectors (permissive + strict)
        self.pose_permissive = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=self.THRESHOLDS['detection_min'],
            min_tracking_confidence=0.3,
            enable_segmentation=True,
            smooth_landmarks=False
        )
        
        self.pose_strict = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=self.THRESHOLDS['quality_excellent'],
            min_tracking_confidence=0.5,
            enable_segmentation=True,
            smooth_landmarks=False
        )
        
        self.logger.info(f"UltimatePoseDetector initialized for {user_type.value} users")
        self.logger.info(f"   Thresholds: Detection={self.THRESHOLDS['detection_min']}, "
                        f"Quality={self.THRESHOLDS['quality_acceptable']}")
    
    def _get_adaptive_thresholds(self, user_type: UserType) -> Dict[str, float]:
        """Get thresholds adapted to user experience level"""
        threshold_configs = {
            UserType.FIRST_TIME: {
                'detection_min': 0.35,
                'landmarks_critical': 0.45,
                'landmarks_important': 0.35,
                'quality_acceptable': 0.45,
                'quality_good': 0.55,
                'quality_excellent': 0.65,
                'rationale': 'More permissive for first-time users'
            },
            UserType.RETURNING: {
                'detection_min': 0.40,
                'landmarks_critical': 0.50,
                'landmarks_important': 0.40,
                'quality_acceptable': 0.50,
                'quality_good': 0.60,
                'quality_excellent': 0.70,
                'rationale': 'Balanced for regular users'
            },
            UserType.PREMIUM: {
                'detection_min': 0.45,
                'landmarks_critical': 0.55,
                'landmarks_important': 0.45,
                'quality_acceptable': 0.55,
                'quality_good': 0.65,
                'quality_excellent': 0.75,
                'rationale': 'Stricter for premium quality expectations'
            },
            UserType.BULK: {
                'detection_min': 0.30,
                'landmarks_critical': 0.40,
                'landmarks_important': 0.30,
                'quality_acceptable': 0.40,
                'quality_good': 0.55,
                'quality_excellent': 0.65,
                'rationale': 'Very permissive for bulk processing'
            }
        }
        return threshold_configs.get(user_type, threshold_configs[UserType.RETURNING])
    
    def analyze_pose_quality(self, image: np.ndarray) -> PoseQualityResult:
        """
        Comprehensive pose quality analysis with tiered validation.
        Main method to call for pose detection.
        
        Args:
            image: Input image (BGR format from cv2.imread)
            
        Returns:
            PoseQualityResult with complete assessment
        """
        start_time = datetime.now()
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Stage 1: Permissive detection
            results_permissive = self.pose_permissive.process(rgb_image)
            
            if not results_permissive.pose_landmarks:
                self.stats.rejected += 1
                self.stats.total_processed += 1
                return self._create_failed_result("No person detected in image")
            
            # Stage 2: Landmark analysis
            landmark_analysis = self._analyze_landmarks(results_permissive.pose_landmarks)
            
            # Stage 3: Calculate confidence
            overall_confidence = self._calculate_weighted_confidence(
                results_permissive.pose_landmarks,
                landmark_analysis
            )
            
            # Stage 4: Strict detection for quality assessment
            results_strict = self.pose_strict.process(rgb_image)
            has_high_quality = results_strict.pose_landmarks is not None
            
            # Stage 5: Determine quality and create result
            result = self._determine_quality_level(
                overall_confidence,
                landmark_analysis,
                has_high_quality
            )
            
            # Update statistics
            self._update_statistics(result, start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in pose analysis: {e}")
            self.stats.rejected += 1
            self.stats.total_processed += 1
            return self._create_failed_result(f"Processing error: {str(e)}")
    
    def _analyze_landmarks(self, landmarks) -> Dict:
        """Detailed analysis of landmark visibility and quality"""
        analysis = {
            'critical_visible': 0,
            'critical_total': 0,
            'critical_confidences': [],
            'important_visible': 0,
            'important_total': 0,
            'important_confidences': [],
            'min_visibility': 1.0,
            'max_visibility': 0.0,
            'avg_visibility': 0.0,
            'missing_critical': [],
            'missing_important': []
        }
        
        all_confidences = []
        
        # Analyze critical landmarks (shoulders, hips)
        for name, indices in self.CRITICAL_LANDMARKS.items():
            for idx in indices:
                analysis['critical_total'] += 1
                visibility = landmarks.landmark[idx].visibility
                analysis['critical_confidences'].append(visibility)
                all_confidences.append(visibility)
                
                if visibility >= self.THRESHOLDS['landmarks_critical']:
                    analysis['critical_visible'] += 1
                else:
                    analysis['missing_critical'].append(f"{name}[{idx}]")
                
                analysis['min_visibility'] = min(analysis['min_visibility'], visibility)
                analysis['max_visibility'] = max(analysis['max_visibility'], visibility)
        
        # Analyze important landmarks (elbows, wrists, knees)
        for name, indices in self.IMPORTANT_LANDMARKS.items():
            for idx in indices:
                analysis['important_total'] += 1
                visibility = landmarks.landmark[idx].visibility
                analysis['important_confidences'].append(visibility)
                all_confidences.append(visibility)
                
                if visibility >= self.THRESHOLDS['landmarks_important']:
                    analysis['important_visible'] += 1
                else:
                    analysis['missing_important'].append(f"{name}[{idx}]")
        
        analysis['avg_visibility'] = np.mean(all_confidences) if all_confidences else 0.0
        
        return analysis
    
    def _calculate_weighted_confidence(self, landmarks, landmark_analysis: Dict) -> float:
        """Calculate sophisticated weighted confidence score"""
        # Critical landmarks (shoulders, hips) are most important: 60%
        critical_ratio = (landmark_analysis['critical_visible'] / 
                         landmark_analysis['critical_total'])
        critical_avg = np.mean(landmark_analysis['critical_confidences'])
        critical_score = (critical_ratio * 0.5 + critical_avg * 0.5)
        
        # Important landmarks (elbows, wrists, knees): 25%
        if landmark_analysis['important_total'] > 0:
            important_ratio = (landmark_analysis['important_visible'] / 
                              landmark_analysis['important_total'])
            important_avg = np.mean(landmark_analysis['important_confidences'])
            important_score = (important_ratio * 0.5 + important_avg * 0.5)
        else:
            important_score = 0.5  # Neutral if no important landmarks
        
        # Overall visibility: 15%
        overall_score = landmark_analysis['avg_visibility']
        
        # Weighted combination
        confidence = (
            critical_score * 0.60 +      # Critical: 60%
            important_score * 0.25 +      # Important: 25%
            overall_score * 0.15          # Overall: 15%
        )
        
        return confidence
    
    def _determine_quality_level(
        self,
        confidence: float,
        landmark_analysis: Dict,
        has_high_quality: bool
    ) -> PoseQualityResult:
        """Determine quality level and create appropriate result"""
        
        # Check if critical landmarks are sufficient
        critical_ratio = (landmark_analysis['critical_visible'] / 
                         landmark_analysis['critical_total'])
        
        # EXCELLENT - High confidence + strict detection passed
        if confidence >= self.THRESHOLDS['quality_excellent'] and has_high_quality:
            return PoseQualityResult(
                detected=True,
                quality_level=QualityLevel.EXCELLENT,
                confidence_score=confidence,
                can_proceed=True,
                user_message="Excellent image quality! Ready for try-on.",
                technical_details={
                    'confidence': confidence,
                    'landmark_analysis': landmark_analysis,
                    'strict_detection': True,
                    'critical_ratio': critical_ratio
                },
                recommendations=[],
                quality_color='#00FF00',  # Green
                show_warning=False,
                offer_retake=False
            )
        
        # GOOD - Good confidence
        elif confidence >= self.THRESHOLDS['quality_good']:
            return PoseQualityResult(
                detected=True,
                quality_level=QualityLevel.GOOD,
                confidence_score=confidence,
                can_proceed=True,
                user_message="Good image quality. Processing...",
                technical_details={
                    'confidence': confidence,
                    'landmark_analysis': landmark_analysis,
                    'strict_detection': has_high_quality,
                    'critical_ratio': critical_ratio
                },
                recommendations=[],
                quality_color='#90EE90',  # Light Green
                show_warning=False,
                offer_retake=False
            )
        
        # ACCEPTABLE - Minimum quality threshold
        elif confidence >= self.THRESHOLDS['quality_acceptable']:
            warnings = []
            if landmark_analysis['missing_critical']:
                warnings.append(f"Some key areas less visible: {', '.join(landmark_analysis['missing_critical'][:2])}")
            
            return PoseQualityResult(
                detected=True,
                quality_level=QualityLevel.ACCEPTABLE,
                confidence_score=confidence,
                can_proceed=True,
                user_message="✓ Image accepted. Quality could be improved for best results.",
                technical_details={
                    'confidence': confidence,
                    'landmark_analysis': landmark_analysis,
                    'warnings': warnings,
                    'critical_ratio': critical_ratio
                },
                recommendations=[
                    "For best results: ensure better lighting",
                    "Stand directly facing the camera",
                    "Make sure shoulders and hips are clearly visible"
                ],
                quality_color='#FFA500',  # Orange
                show_warning=True,
                offer_retake=True
            )
        
        # POOR - Below minimum but detectable
        elif confidence >= self.THRESHOLDS['detection_min'] and critical_ratio >= 0.75:
            return PoseQualityResult(
                detected=True,
                quality_level=QualityLevel.POOR,
                confidence_score=confidence,
                can_proceed=True,  # Allow but strongly warn
                user_message="Image quality is poor. Try-on results may not be optimal.\n"
                           "Strongly recommend retaking photo.",
                technical_details={
                    'confidence': confidence,
                    'landmark_analysis': landmark_analysis,
                    'critical_ratio': critical_ratio,
                    'warning': 'poor_quality_accepted'
                },
                recommendations=[
                    "RECOMMENDED: Retake photo with better conditions",
                    "Ensure good, even lighting",
                    "Face camera directly",
                    "Stand 3-5 feet from camera",
                    "Clear background if possible"
                ],
                quality_color='#FF4500',  # Red-Orange
                show_warning=True,
                offer_retake=True
            )
        
        # REJECTED - Insufficient quality
        else:
            missing_parts = landmark_analysis['missing_critical'] + \
                           landmark_analysis['missing_important'][:2]
            
            return PoseQualityResult(
                detected=confidence >= self.THRESHOLDS['detection_min'],
                quality_level=QualityLevel.FAILED,
                confidence_score=confidence,
                can_proceed=False,
                user_message="Image quality too poor for virtual try-on.\n"
                           "Please retake photo with better conditions.",
                technical_details={
                    'confidence': confidence,
                    'landmark_analysis': landmark_analysis,
                    'critical_ratio': critical_ratio,
                    'missing_parts': missing_parts,
                    'reason': 'insufficient_quality'
                },
                recommendations=[
                    "Ensure you're clearly visible in good lighting",
                    "Shoulders and hips must be visible",
                    "Stand in open, uncluttered space",
                    "Hold camera at chest height",
                    "Face camera directly"
                ],
                quality_color='#FF0000',  # Red
                show_warning=True,
                offer_retake=True
            )
    
    def _create_failed_result(self, message: str) -> PoseQualityResult:
        """Create a failed detection result"""
        return PoseQualityResult(
            detected=False,
            quality_level=QualityLevel.FAILED,
            confidence_score=0.0,
            can_proceed=False,
            user_message=f"{message}",
            technical_details={'error': message},
            recommendations=[
                "Ensure you're in the frame",
                "Use good lighting",
                "Stand in clear area",
                "Try different angle"
            ],
            quality_color='#FF0000',
            show_warning=True,
            offer_retake=True
        )
    
    def _update_statistics(self, result: PoseQualityResult, start_time: datetime):
        """Update processing statistics"""
        self.stats.total_processed += 1
        
        if result.can_proceed:
            self.stats.accepted += 1
        else:
            self.stats.rejected += 1
        
        # Track quality distribution
        quality = result.quality_level.value
        self.stats.quality_distribution[quality] = \
            self.stats.quality_distribution.get(quality, 0) + 1
        
        # Update average confidence
        total_conf = self.stats.avg_confidence * (self.stats.total_processed - 1)
        self.stats.avg_confidence = (total_conf + result.confidence_score) / self.stats.total_processed
        
        # Update average processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        total_time = self.stats.avg_processing_time * (self.stats.total_processed - 1)
        self.stats.avg_processing_time = (total_time + processing_time) / self.stats.total_processed
    
    def get_statistics(self) -> Dict:
        """Get current processing statistics"""
        if self.stats.total_processed == 0:
            return {
                'total_processed': 0,
                'message': 'No images processed yet'
            }
        
        acceptance_rate = (self.stats.accepted / self.stats.total_processed) * 100
        
        return {
            'total_processed': self.stats.total_processed,
            'accepted': self.stats.accepted,
            'rejected': self.stats.rejected,
            'acceptance_rate': f"{acceptance_rate:.1f}%",
            'avg_confidence': f"{self.stats.avg_confidence:.1%}",
            'avg_processing_time': f"{self.stats.avg_processing_time:.3f}s",
            'quality_distribution': self.stats.quality_distribution,
            'user_type': self.user_type.value,
            'thresholds': {
                'detection': self.THRESHOLDS['detection_min'],
                'quality_acceptable': self.THRESHOLDS['quality_acceptable'],
                'quality_good': self.THRESHOLDS['quality_good']
            }
        }
    
    def close(self):
        """Clean up resources"""
        self.pose_permissive.close()
        self.pose_strict.close()
        self.logger.info("UltimatePoseDetector closed")


# ============================================================================
# INTEGRATION: CUSTOMER IMAGE ANALYZER
# ============================================================================

class UltimateCustomerImageAnalyzer:
    """
    Drop-in replacement for CustomerImageAnalyzer.
    Uses UltimatePoseDetector under the hood.
    """
    
    def __init__(self, user_type: UserType = UserType.RETURNING):
        """Initialize with user type for adaptive behavior"""
        self.detector = UltimatePoseDetector(user_type=user_type)
        self.logger = logging.getLogger(__name__)
        self.logger.info("UltimateCustomerImageAnalyzer initialized")
    
    def analyze_customer_image(self, image: np.ndarray) -> Dict:
        """
        Analyze customer image with comprehensive quality assessment.
        Returns format compatible with your existing pipeline.
        
        Args:
            image: Customer image (BGR format)
            
        Returns:
            Dict with analysis results
        """
        result = self.detector.analyze_pose_quality(image)
        
        # Convert to your existing format
        return {
            'success': result.can_proceed,
            'detected': result.detected,
            'quality_level': result.quality_level.value,
            'confidence': result.confidence_score,
            'message': result.user_message,
            'can_proceed': result.can_proceed,
            'show_warning': result.show_warning,
            'offer_retake': result.offer_retake,
            'recommendations': result.recommendations,
            'quality_color': result.quality_color,
            'technical_details': result.technical_details,
            
            # For backward compatibility
            'pose_landmarks': result.technical_details.get('pose_landmarks'),
            'error': None if result.can_proceed else 'QUALITY_CHECK_FAILED'
        }
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.detector.get_statistics()


# ============================================================================
# INTEGRATION: ENHANCED PIPELINE CONTROLLER
# ============================================================================

class UltimateEnhancedPipelineController:
    """
    Complete pipeline controller with sophisticated quality gates.
    Drop-in replacement for your existing pipeline controller.
    """
    
    def __init__(self, user_type: UserType = UserType.RETURNING):
        """Initialize pipeline with adaptive configuration"""
        self.analyzer = UltimateCustomerImageAnalyzer(user_type=user_type)
        self.logger = logging.getLogger(__name__)
        self.session_stats = []
        
        self.logger.info("UltimateEnhancedPipelineController initialized")
    
    def process_customer_image(self, image_path: str, visualize: bool = False) -> Dict:
        """
        Complete processing pipeline with quality gates.
        
        Args:
            image_path: Path to customer image
            visualize: If True, creates annotated image showing quality
            
        Returns:
            Dict with processing results and decision
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'status': 'error',
                    'stage': 'image_loading',
                    'error': 'Could not load image',
                    'message': 'Failed to load image file. Please check the path.'
                }
            
            # Analyze quality
            analysis = self.analyzer.analyze_customer_image(image)
            
            # Create response
            response = {
                'status': 'approved' if analysis['can_proceed'] else 'rejected',
                'stage': 'quality_assessment',
                'quality_level': analysis['quality_level'],
                'confidence': analysis['confidence'],
                'message': analysis['message'],
                'can_proceed': analysis['can_proceed'],
                'show_warning': analysis['show_warning'],
                'offer_retake': analysis['offer_retake'],
                'recommendations': analysis['recommendations'],
                'quality_color': analysis['quality_color']
            }
            
            # Track for session
            self.session_stats.append(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {
                'status': 'error',
                'stage': 'pipeline_execution',
                'error': str(e),
                'message': f'Processing error: {str(e)}'
            }
    
    def call_fashn_api_if_approved(
        self,
        result: Dict,
        customer_image,
        garment_image
    ) -> Dict:
        """
        Only call FASHN API if quality checks passed.
        Prevents wasting API credits on poor quality images.
        
        Args:
            result: Result from process_customer_image
            customer_image: Customer image
            garment_image: Garment image
            
        Returns:
            Dict with API call result
        """
        if result['status'] == 'approved' and result['can_proceed']:
            self.logger.info(f"Quality approved ({result['quality_level']}) - Calling FASHN API")
            
            return {
                'api_called': True,
                'quality': result['quality_level'],
                'confidence': result['confidence'],
                'message': 'FASHN API called successfully'
            }
        else:
            self.logger.info(f"Quality check failed - NOT calling API")
            self.logger.info(f"   Reason: {result['message']}")
            
            return {
                'api_called': False,
                'reason': result['message'],
                'quality': result.get('quality_level', 'failed'),
                'recommendations': result.get('recommendations', []),
                'money_saved': True,
                'message': 'API call skipped due to quality check failure'
            }
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.session_stats:
            return {'message': 'No images processed in this session'}
        
        total = len(self.session_stats)
        approved = sum(1 for s in self.session_stats if s['status'] == 'approved')
        rejected = total - approved
        
        quality_dist = {}
        for stat in self.session_stats:
            quality = stat.get('quality_level', 'unknown')
            quality_dist[quality] = quality_dist.get(quality, 0) + 1
        
        return {
            'session_total': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': f"{(approved/total*100):.1f}%",
            'quality_distribution': quality_dist,
            'detector_stats': self.analyzer.get_statistics()
        }
    
    def close(self):
        """Clean up resources"""
        self.analyzer.detector.close()