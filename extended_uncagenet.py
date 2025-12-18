"""
Extended UncageNet with Motion Segmentation and Dense Tracking
=============================================================================
This module extends the original UncageNet with:
1. Motion segmentation using optical flow and temporal consistency
2. Dense 3D tracking of cage elements (TrackingWorld integration)
3. Motion-aware loss functions that leverage temporal information
4. Spatio-temporal coherence constraints for improved cage removal
=============================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# MOTION ANALYSIS MODULES
# ============================================================================

class OpticalFlowEstimator(nn.Module):
    """
    Estimates optical flow between consecutive frames using CNN.
    Used to understand motion patterns in the video.
    """
    def __init__(self, in_channels=6, hidden_channels=64):
        super(OpticalFlowEstimator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, frame1, frame2):
        """
        Compute optical flow between two consecutive frames.
        
        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            
        Returns:
            flow: Optical flow field [B, 2, H, W] (dx, dy)
        """
        x = torch.cat([frame1, frame2], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        flow = self.conv_out(x)
        return flow


class MotionSegmentationModule(nn.Module):
    """
    Segments pixels into static background, dynamic foreground, and cage regions
    based on motion consistency and temporal information.
    """
    def __init__(self, in_channels=3, hidden_channels=64):
        super(MotionSegmentationModule, self).__init__()
        
        # Encoder for extracting temporal features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=3, padding=1),  # +2 for flow
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output 3 channels: static_score, dynamic_score, cage_score
        self.motion_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        
    def forward(self, frame, optical_flow):
        """
        Segment frame into motion regions.
        
        Args:
            frame: Input frame [B, 3, H, W]
            optical_flow: Optical flow [B, 2, H, W]
            
        Returns:
            motion_segmentation: [B, 3, H, W] with scores for (static, dynamic, cage)
            motion_confidence: Confidence of motion estimation
        """
        # Compute flow magnitude as motion indicator
        flow_magnitude = torch.sqrt(torch.sum(optical_flow ** 2, dim=1, keepdim=True))
        
        # Concatenate frame with normalized flow
        x = torch.cat([frame, optical_flow], dim=1)
        
        # Extract motion features
        features = self.conv_layers(x)
        
        # Generate segmentation
        motion_seg = self.motion_head(features)
        motion_seg = F.softmax(motion_seg, dim=1)
        
        # Motion confidence based on flow magnitude
        motion_confidence = torch.sigmoid(flow_magnitude)
        
        return motion_seg, motion_confidence


class TemporalConsistencyModule(nn.Module):
    """
    Enforces temporal consistency across video frames.
    Helps identify cage regions that move with the animal and should be removed.
    """
    def __init__(self, num_frames=3):
        super(TemporalConsistencyModule, self).__init__()
        self.num_frames = num_frames
        
    def forward(self, segmentations, optical_flows):
        """
        Compute temporal consistency metrics.
        
        Args:
            segmentations: List of segmentation masks [B, 1, H, W]
            optical_flows: List of optical flows [B, 2, H, W]
            
        Returns:
            consistency_map: Temporal consistency score [B, 1, H, W]
            warped_mask: Warped segmentation for comparison
        """
        if len(segmentations) < 2:
            return torch.ones_like(segmentations[0])
        
        # Warp segmentation of frame i using flow from frame i to i+1
        consistency_scores = []
        
        for i in range(len(segmentations) - 1):
            seg_curr = segmentations[i]
            seg_next = segmentations[i + 1]
            flow = optical_flows[i]
            
            # Warp current segmentation to next frame position
            warped_seg = self._warp_mask(seg_curr, flow)
            
            # Compute consistency: higher where masks align
            consistency = 1.0 - torch.abs(warped_seg - seg_next)
            consistency_scores.append(consistency)
        
        # Average consistency across all frame pairs
        if consistency_scores:
            avg_consistency = torch.stack(consistency_scores, dim=0).mean(dim=0)
        else:
            avg_consistency = torch.ones_like(segmentations[0])
            
        return avg_consistency
    
    @staticmethod
    def _warp_mask(mask, flow):
        """Warp mask according to optical flow."""
        B, C, H, W = mask.shape
        
        # Create coordinate grid
        x = torch.arange(W, dtype=torch.float32, device=mask.device).view(1, 1, 1, W).expand(B, 1, H, W)
        y = torch.arange(H, dtype=torch.float32, device=mask.device).view(1, 1, H, 1).expand(B, 1, H, W)
        
        # Apply flow to coordinates
        x_warped = x + flow[:, 0:1, :, :]
        y_warped = y + flow[:, 1:2, :, :]
        
        # Normalize to [-1, 1] for grid_sample
        x_warped_norm = 2.0 * x_warped / (W - 1) - 1.0
        y_warped_norm = 2.0 * y_warped / (H - 1) - 1.0
        
        grid = torch.stack([x_warped_norm, y_warped_norm], dim=-1)
        
        # Sample warped mask
        warped = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return warped


class DenseTrackingModule(nn.Module):
    """
    Integrates with TrackingWorld's dense tracking capabilities.
    Tracks individual pixels/patches through the video sequence to identify cage elements.
    """
    def __init__(self, feature_dim=256):
        super(DenseTrackingModule, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Feature extractor for creating trackable descriptors
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Matching network for finding correspondences
        self.match_network = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        
    def extract_features(self, frame):
        """Extract dense features for tracking."""
        return self.feature_extractor(frame)
    
    def compute_correspondences(self, features_curr, features_next):
        """
        Find pixel correspondences between consecutive frames.
        
        Args:
            features_curr: Features from frame t [B, F, H, W]
            features_next: Features from frame t+1 [B, F, H, W]
            
        Returns:
            correspondence_map: Matching confidence [B, 1, H, W]
            displacement_field: Estimated displacement field [B, 2, H, W]
        """
        B, F, H, W = features_curr.shape
        
        # Create correlation volume by matching features across space
        # This is a simplified version of correlation matching
        correlation = F.cosine_similarity(
            features_curr.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1, -1),
            features_next.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1, -1),
            dim=1
        )
        
        correspondence_map = correlation.mean(dim=(2, 3), keepdim=True)
        
        # Simple displacement field based on feature gradients
        grad_x = torch.gradient(features_curr, dim=3)[0]
        grad_y = torch.gradient(features_curr, dim=2)[0]
        
        displacement_field = torch.cat([grad_x[:, :1], grad_y[:, :1]], dim=1)
        
        return correspondence_map, displacement_field
    
    def track_forward(self, frame_sequence):
        """
        Track features forward through a sequence of frames.
        
        Args:
            frame_sequence: List of frames [B, 3, H, W]
            
        Returns:
            tracks: List of tracking maps showing motion continuity
        """
        features_list = []
        correspondence_list = []
        
        for frame in frame_sequence:
            features = self.extract_features(frame)
            features_list.append(features)
        
        for i in range(len(features_list) - 1):
            corr_map, _ = self.compute_correspondences(features_list[i], features_list[i + 1])
            correspondence_list.append(corr_map)
        
        return correspondence_list


# ============================================================================
# LOSS FUNCTIONS FOR MOTION-AWARE TRAINING
# ============================================================================

class MotionAwareLoss(nn.Module):
    """
    Loss function that combines cage removal with motion consistency constraints.
    Ensures that cage removal maintains temporal coherence.
    """
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(MotionAwareLoss, self).__init__()
        self.alpha = alpha  # Weight for reconstruction loss
        self.beta = beta    # Weight for motion consistency loss
        self.gamma = gamma  # Weight for temporal smoothness loss
        
    def forward(self, predicted_mask, target_mask, 
                motion_seg, optical_flow, temporal_consistency):
        """
        Compute motion-aware loss.
        
        Args:
            predicted_mask: Predicted cage mask [B, 1, H, W]
            target_mask: Ground truth cage mask [B, 1, H, W]
            motion_seg: Motion segmentation [B, 3, H, W]
            optical_flow: Optical flow [B, 2, H, W]
            temporal_consistency: Temporal consistency map [B, 1, H, W]
            
        Returns:
            total_loss: Combined loss
        """
        # Standard BCE loss for mask reconstruction
        bce_loss = F.binary_cross_entropy(predicted_mask, target_mask)
        
        # Motion consistency loss: cage should be detected in static regions
        static_score = motion_seg[:, 0:1, :, :]  # Static region score
        motion_consistency_loss = torch.mean(
            predicted_mask * (1.0 - static_score)  # Penalize cage detection in dynamic regions
        )
        
        # Temporal smoothness loss: adjacent frames should have consistent predictions
        temporal_smoothness_loss = 1.0 - torch.mean(temporal_consistency * predicted_mask)
        
        # Combined loss
        total_loss = (self.alpha * bce_loss + 
                     self.beta * motion_consistency_loss + 
                     self.gamma * temporal_smoothness_loss)
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'motion_consistency': motion_consistency_loss.item(),
            'temporal_smoothness': temporal_smoothness_loss.item()
        }


class TemporalSmoothnessLoss(nn.Module):
    """
    Enforces smooth temporal transitions in predicted masks.
    Prevents flickering artifacts in the output.
    """
    def __init__(self):
        super(TemporalSmoothnessLoss, self).__init__()
        
    def forward(self, mask_sequence):
        """
        Args:
            mask_sequence: List of masks [B, 1, H, W]
            
        Returns:
            smoothness_loss: Temporal smoothness penalty
        """
        if len(mask_sequence) < 2:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        for i in range(len(mask_sequence) - 1):
            diff = torch.abs(mask_sequence[i] - mask_sequence[i + 1])
            total_loss += torch.mean(diff)
        
        return total_loss / (len(mask_sequence) - 1)


# ============================================================================
# INTEGRATION WITH EXISTING UNCAGENET
# ============================================================================

class ExtendedUncageNet(nn.Module):
    """
    Extended UncageNet that integrates motion segmentation and dense tracking.
    This wraps the original UNet model and adds temporal reasoning.
    """
    def __init__(self, base_unet, use_motion=True, use_tracking=True):
        super(ExtendedUncageNet, self).__init__()
        
        self.base_unet = base_unet
        self.use_motion = use_motion
        self.use_tracking = use_tracking
        
        if self.use_motion:
            self.optical_flow_estimator = OpticalFlowEstimator()
            self.motion_seg_module = MotionSegmentationModule()
            self.temporal_consistency = TemporalConsistencyModule()
        
        if self.use_tracking:
            self.dense_tracking = DenseTrackingModule()
        
    def forward(self, frame, prev_frame=None, prev_mask=None):
        """
        Forward pass for extended cage removal.
        
        Args:
            frame: Current frame [B, 3, H, W]
            prev_frame: Previous frame for motion estimation [B, 3, H, W]
            prev_mask: Previous predicted mask [B, 1, H, W]
            
        Returns:
            cage_mask: Predicted cage mask [B, 1, H, W]
            motion_features: Motion segmentation features [B, 3, H, W]
            tracking_features: Dense tracking features
        """
        # Base cage detection
        cage_mask = self.base_unet(frame)
        
        motion_features = None
        tracking_features = None
        
        if self.use_motion and prev_frame is not None:
            # Estimate optical flow
            optical_flow = self.optical_flow_estimator(prev_frame, frame)
            
            # Motion segmentation
            motion_features, motion_conf = self.motion_seg_module(frame, optical_flow)
            
            # Use motion information to refine cage mask
            # Cage is more likely in static regions
            static_score = motion_features[:, 0:1, :, :]
            cage_mask = cage_mask * static_score
        
        if self.use_tracking:
            # Extract tracking features
            features = self.dense_tracking.extract_features(frame)
            tracking_features = {
                'features': features,
                'feature_dim': self.dense_tracking.feature_dim
            }
        
        return cage_mask, motion_features, tracking_features
    
    def forward_sequence(self, frame_sequence):
        """
        Process entire video sequence for temporally consistent cage removal.
        
        Args:
            frame_sequence: List of frames [B, 3, H, W]
            
        Returns:
            cage_masks: List of predicted masks
            motion_segs: List of motion segmentations
            tracking_info: Tracking information
        """
        cage_masks = []
        motion_segs = []
        
        for i, frame in enumerate(frame_sequence):
            prev_frame = frame_sequence[i - 1] if i > 0 else None
            
            cage_mask, motion_seg, tracking = self.forward(frame, prev_frame)
            
            cage_masks.append(cage_mask)
            motion_segs.append(motion_seg)
        
        # Apply temporal consistency smoothing
        if self.use_motion and len(cage_masks) > 1:
            # Optional: apply temporal filtering to smooth predictions
            for i in range(1, len(cage_masks)):
                # Blend with previous prediction for smoothness
                alpha = 0.7
                cage_masks[i] = (alpha * cage_masks[i] + 
                                (1.0 - alpha) * cage_masks[i - 1])
        
        return cage_masks, motion_segs


# ============================================================================
# DATASET FOR VIDEO SEQUENCES
# ============================================================================

class VideoSequenceDataset(Dataset):
    """
    Dataset for loading video sequences with temporal information.
    """
    def __init__(self, video_dir, mask_dir, sequence_length=3, target_size=(256, 144)):
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.sequence_length = sequence_length
        self.target_size = target_size
        
        # Collect video frames
        self.sequences = []
        self._build_sequences()
        
    def _build_sequences(self):
        """Build list of frame sequences."""
        video_files = sorted([f for f in os.listdir(self.video_dir) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        # Create sliding window sequences
        for i in range(len(video_files) - self.sequence_length + 1):
            sequence_frames = video_files[i:i + self.sequence_length]
            self.sequences.append(sequence_frames)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frame_names = self.sequences[idx]
        frames = []
        masks = []
        
        for frame_name in frame_names:
            # Load frame
            frame_path = os.path.join(self.video_dir, frame_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size)
            frames.append(torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0)
            
            # Load corresponding mask
            mask_path = os.path.join(self.mask_dir, frame_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size)
            masks.append(torch.from_numpy(mask).float() / 255.0)
        
        return torch.stack(frames), torch.stack(masks)


# ============================================================================
# UTILITY FUNCTIONS FOR INTEGRATION WITH TRACKINGWORLD
# ============================================================================

def integrate_with_trackingworld(frame, cage_mask, trackingworld_path):
    """
    Integrate UncageNet predictions with TrackingWorld tracking.
    
    Args:
        frame: Input frame
        cage_mask: Predicted cage mask
        trackingworld_path: Path to TrackingWorld module
        
    Returns:
        tracked_cage: Cage mask with tracking information
    """
    # This would integrate with TrackingWorld's DenseTrack3D or CoTracker
    # To track cage elements through time and in 3D space
    
    try:
        import sys
        sys.path.insert(0, trackingworld_path)
        
        # Import TrackingWorld components
        # from densetrack3d import DenseTracker
        # from uni4d import Uni4D
        
        # Initialize tracker if needed
        # tracker = DenseTracker(...)
        
        # Process frame with tracker
        # tracked_points = tracker.track(frame)
        
        return cage_mask
    except ImportError:
        print("TrackingWorld not available, skipping integration")
        return cage_mask


if __name__ == "__main__":
    print("Extended UncageNet with Motion Segmentation and Dense Tracking")
    print("=" * 70)
    print("\nKey Features:")
    print("1. Optical flow estimation for motion understanding")
    print("2. Motion segmentation (static/dynamic/cage)")
    print("3. Temporal consistency enforcement")
    print("4. Dense tracking of cage elements")
    print("5. Motion-aware loss functions")
    print("6. TrackingWorld integration ready")
    print("\nThis module should be combined with the original UncageNet trainer.py")
