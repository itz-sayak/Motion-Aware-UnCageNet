"""
Motion-Aware UncageNet Model with ResNet Backbone
==================================================
Extended UncageNet with ResNet encoder, motion segmentation and dense tracking integration.
This module provides the neural network architecture for motion-aware cage removal.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

The Motion-Aware UncageNet is a U-Net style encoder-decoder network enhanced with:
1. ResNet101 backbone (ImageNet pretrained)
2. Gabor filter preprocessing for orientation features
3. Optional motion/optical flow integration
4. Attention gates in skip connections

================================================================================
FULL PIPELINE BLOCK DIAGRAM
================================================================================

                            INPUT PIPELINE
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   ┌─────────────┐                                                    │
    │   │  RGB Image  │──────────────────────────────────────┐             │
    │   │  [B,3,H,W]  │                                      │             │
    │   └─────────────┘                                      │             │
    │         │                                              │             │
    │         ▼                                              ▼             │
    │   ┌─────────────┐     ┌─────────────┐          ┌─────────────┐       │
    │   │  Grayscale  │────▶│Gabor Filter │─────────▶│  Orientation│       │
    │   │  [B,1,H,W]  │     │ (72 kernels)│          │  [B,2,H,W]  │       │
    │   └─────────────┘     └─────────────┘          │ (sin,cos)   │       │
    │                             │                   └──────┬──────┘       │
    │                             ▼                          │             │
    │                       ┌───────────┐                    │             │
    │                       │Confidence │                    │             │
    │                       │  Masking  │                    │             │
    │                       └───────────┘                    │             │
    │                                                        │             │
    │   ┌─────────────┐     ┌─────────────┐                  │             │
    │   │Optical Flow │────▶│Flow Encoder │──────────┐       │             │
    │   │  [B,2,H,W]  │     │ Conv 2→32   │          │       │             │
    │   │ (optional)  │     │ Conv 32→32  │          │       │             │
    │   └─────────────┘     └─────────────┘          │       │             │
    │                             │                  │       │             │
    │                             ▼                  │       │             │
    │                       ┌───────────┐            │       │             │
    │                       │  Motion   │            │       │             │
    │                       │Segmentation│───────────┼───────┼─────▶ Aux   │
    │                       │  Head     │            │       │      Output │
    │                       │[static,   │            │       │             │
    │                       │ dynamic,  │            │       │             │
    │                       │ cage]     │            │       │             │
    │                       └───────────┘            │       │             │
    │                                                │       │             │
    │                       ┌────────────────────────┴───────┴─────┐       │
    │                       │           CONCATENATE                │       │
    │                       │  [RGB + Gabor + Flow] = [B,5+,H,W]   │       │
    │                       └──────────────────┬───────────────────┘       │
    │                                          │                           │
    └──────────────────────────────────────────┼───────────────────────────┘
                                               │
                                               ▼
================================================================================
                            RESNET ENCODER (Backbone)
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   Input: [B, 5, H, W] (RGB + Gabor orientation channels)             │
    │                                                                      │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Modified Conv1: 5→64, 7x7, stride=2                         │    │
    │   │ BatchNorm + ReLU                                            │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ f1: [B, 64, H/2, W/2]                 │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ MaxPool: 3x3, stride=2                                      │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Layer1 (ResNet101): 3 Bottleneck blocks                     │    │
    │   │ 64 → 256 channels                                           │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ f2: [B, 256, H/4, W/4]                │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Layer2 (ResNet101): 4 Bottleneck blocks                     │    │
    │   │ 256 → 512 channels, stride=2                                │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ f3: [B, 512, H/8, W/8]                │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Layer3 (ResNet101): 23 Bottleneck blocks                    │    │
    │   │ 512 → 1024 channels, stride=2                               │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ f4: [B, 1024, H/16, W/16]             │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Layer4 (ResNet101): 3 Bottleneck blocks                     │    │
    │   │ 1024 → 2048 channels, stride=2                              │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ f5: [B, 2048, H/32, W/32]             │
    │                              ▼                                       │
    │   Skip connections: [f1, f2, f3, f4, f5]                             │
    │                                                                      │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
================================================================================
                            CENTER BLOCK + TEMPORAL FUSION
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ DoubleConv: 2048 → 1024                                     │    │
    │   │ Conv 3x3 + BN + ReLU + Conv 3x3 + BN + ReLU                 │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Temporal Fusion Module (if use_motion)                      │    │
    │   │ ┌─────────────────────────────────────────────────────┐     │    │
    │   │ │  current_features ─┬─▶ Concat ─▶ Conv ─▶ Attention   │     │    │
    │   │ │  prev_features ────┘            weights              │     │    │
    │   │ │                                    │                 │     │    │
    │   │ │  fused = curr * att + prev * (1-att)                 │     │    │
    │   │ └─────────────────────────────────────────────────────┘     │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ bottleneck: [B, 1024, H/32, W/32]     │
    │                              ▼                                       │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
================================================================================
                            DECODER WITH ATTENTION GATES
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   For each decoder stage:                                            │
    │                                                                      │
    │   ┌───────────────────────────────────────────────────────────────┐  │
    │   │                    ATTENTION GATE                             │  │
    │   │  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │  │
    │   │  │  Gate g  │───▶│ W_g(g)   │───▶│          │                 │  │
    │   │  │(decoder) │    │ Conv 1x1 │    │   ADD    │                 │  │
    │   │  └──────────┘    └──────────┘    │    +     │                 │  │
    │   │                                  │   ReLU   │                 │  │
    │   │  ┌──────────┐    ┌──────────┐    │    ▼     │    ┌─────────┐  │  │
    │   │  │  Skip x  │───▶│ W_x(x)   │───▶│   psi    │───▶│ x * psi │  │  │
    │   │  │(encoder) │    │ Conv 1x1 │    │ sigmoid  │    │(attended│  │  │
    │   │  └──────────┘    └──────────┘    └──────────┘    │  skip)  │  │  │
    │   │                                                  └─────────┘  │  │
    │   └───────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │   DECODER BLOCKS:                                                    │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Dec1: Upsample 2x + Concat(f4_att) + Conv                   │    │
    │   │       [1024 + 1024] → 1024                                  │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ [B, 1024, H/16, W/16]                 │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Dec2: Upsample 2x + Concat(f3_att) + Conv                   │    │
    │   │       [1024 + 512] → 512                                    │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ [B, 512, H/8, W/8]                    │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Dec3: Upsample 2x + Concat(f2_att) + Conv                   │    │
    │   │       [512 + 256] → 256                                     │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ [B, 256, H/4, W/4]                    │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Dec4: Upsample 2x + Concat(f1_att) + Conv                   │    │
    │   │       [256 + 64] → 128                                      │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ [B, 128, H/2, W/2]                    │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Dec5: Upsample 2x + Conv                                    │    │
    │   │       [128] → 64                                            │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │ [B, 64, H, W]                         │
    │                              ▼                                       │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
================================================================================
                            OUTPUT HEAD
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Segmentation Head: Conv 1x1                                 │    │
    │   │ 64 → 1 (logits, no sigmoid - for BCEWithLogitsLoss)         │    │
    │   └──────────────────────────┬──────────────────────────────────┘    │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐    │
    │   │ Output: [B, 1, H, W] - Cage segmentation logits             │    │
    │   │                                                             │    │
    │   │ During inference: Apply sigmoid for probability mask        │    │
    │   └─────────────────────────────────────────────────────────────┘    │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
                            LOSS FUNCTION
================================================================================
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   Total Loss = w_bce * BCE_Loss + w_dice * Dice_Loss                 │
    │              + w_motion * Motion_Loss + w_temporal * Temporal_Loss   │
    │                                                                      │
    │   ┌────────────────────────┐                                         │
    │   │ BCE Loss (on logits)   │  Binary Cross-Entropy with Logits       │
    │   │ BCEWithLogitsLoss      │  (autocast compatible)                  │
    │   └────────────────────────┘                                         │
    │                                                                      │
    │   ┌────────────────────────┐                                         │
    │   │ Dice Loss (on probs)   │  1 - (2*|P∩T|)/(|P|+|T|)                │
    │   │ (after sigmoid)        │  Handles class imbalance                │
    │   └────────────────────────┘                                         │
    │                                                                      │
    │   ┌────────────────────────┐                                         │
    │   │ Motion Loss            │  Penalizes cage predictions in          │
    │   │ (optional)             │  dynamic (non-static) regions           │
    │   └────────────────────────┘                                         │
    │                                                                      │
    │   ┌────────────────────────┐                                         │
    │   │ Temporal Loss          │  L1 consistency between                 │
    │   │ (optional)             │  consecutive frame predictions          │
    │   └────────────────────────┘                                         │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
                            METRICS COMPUTED
================================================================================
    • IoU (Intersection over Union)
    • Dice Score (F1 for segmentation)
    • Precision = TP / (TP + FP)
    • Recall = TP / (TP + FN)
    • Specificity = TN / (TN + FP)
    • Accuracy = (TP + TN) / Total

================================================================================
                            MODEL CONFIGURATIONS
================================================================================
    
    ResNet Backbones Available:
    ┌─────────────┬────────────────────┬──────────────────────────────────┐
    │ Backbone    │ Encoder Channels   │ Parameters (approx)              │
    ├─────────────┼────────────────────┼──────────────────────────────────┤
    │ resnet18    │ [64,64,128,256,512]│ ~15M total                       │
    │ resnet34    │ [64,64,128,256,512]│ ~25M total                       │
    │ resnet50    │ [64,256,512,1024,  │ ~50M total                       │
    │             │  2048]             │                                  │
    │ resnet101   │ [64,256,512,1024,  │ ~113M total (recommended)        │
    │             │  2048]             │                                  │
    └─────────────┴────────────────────┴──────────────────────────────────┘
    
    Optional Features:
    • use_gabor=True     : Adds 2 orientation channels (sin, cos)
    • use_attention=True : Adds attention gates to skip connections
    • use_motion=True    : Adds optical flow processing + temporal fusion

================================================================================
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

# Try to import Gabor filter from trainer.py
try:
    from trainer import calOrientationGabor
    HAS_GABOR = True
except ImportError:
    HAS_GABOR = False
    print("Note: calOrientationGabor not found in trainer.py, using standalone implementation")


# ============================================================================
# GABOR FILTER (if not imported)
# ============================================================================

if not HAS_GABOR:
    class calOrientationGabor(nn.Module):
        """
        Gabor filter module for extracting orientation features.
        """
        def __init__(self, channel_in=1, channel_out=1, stride=1, device=None):
            super(calOrientationGabor, self).__init__()
            self.channel_in = channel_in
            self.channel_out = channel_out
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.numKernels = 72
            self.clamp_confidence_low = 0.0
            self.clamp_confidence_high = 0.2
            
            self.sigma_x = nn.Parameter(torch.ones(1) * 1.8)
            self.sigma_y = nn.Parameter(torch.ones(1) * 2.4)
            self.Lambda = nn.Parameter(torch.ones(1) * 4.0)
            self.kernel_size = 17
            
        def gabor_fn(self, kernel_size, channel_in, channel_out, theta, sigma_x, sigma_y, Lambda, phase=0.):
            sigma_x_expanded = sigma_x.expand(channel_out)
            sigma_y_expanded = sigma_y.expand(channel_out)
            Lambda_expanded = Lambda.expand(channel_out)
            psi = nn.Parameter(torch.ones(channel_out) * phase).to(self.device)

            xmax = kernel_size // 2
            ymax = kernel_size // 2
            xmin = -xmax
            ymin = -ymax
            ksize = xmax - xmin + 1
            y_0 = torch.arange(ymin, ymax + 1).to(self.device).float() - 0.5
            y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
            x_0 = torch.arange(xmin, xmax + 1).to(self.device).float() - 0.5
            x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()

            x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
            y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

            gb = torch.exp(
                -.5 * (x_theta ** 2 / sigma_x_expanded.view(-1, 1, 1, 1) ** 2 + 
                       y_theta ** 2 / sigma_y_expanded.view(-1, 1, 1, 1) ** 2)) \
                 * torch.cos(2 * math.pi * x_theta / Lambda_expanded.view(-1, 1, 1, 1) + psi.view(-1, 1, 1, 1))

            return gb

        def filter(self, image, label, threshold, variance_data, orient_data, max_resp_data):
            batch_size = 6
            H, W = image.size()[2:4]
            
            all_responses = []
            all_orients = []
            
            for batch_start in range(0, self.numKernels, batch_size):
                batch_end = min(batch_start + batch_size, self.numKernels)
                batch_resArray = []
                batch_orients = []
                
                for iOrient in range(batch_start, batch_end):
                    theta = nn.Parameter(torch.ones(self.channel_out) * (math.pi * iOrient / self.numKernels)).to(self.device)
                    GaborKernel = self.gabor_fn(self.kernel_size, self.channel_in, self.channel_out, theta, 
                                                self.sigma_x, self.sigma_y, self.Lambda)
                    
                    response = F.conv2d(image, GaborKernel, padding=self.kernel_size//2)
                    batch_resArray.append(response)
                    
                    orient_tensor = torch.ones(1, 1, H, W).to(self.device) * math.pi * iOrient / self.numKernels
                    batch_orients.append(orient_tensor)
                
                all_responses.extend(batch_resArray)
                all_orients.extend(batch_orients)
                torch.cuda.empty_cache()
            
            resTensor = torch.cat(all_responses, dim=1)
            orient = torch.cat(all_orients, dim=1)
            
            resTensor = torch.abs(resTensor)
            max_resp = torch.max(resTensor, dim=1, keepdim=True)[0]
            maxResTensor = torch.argmax(resTensor, dim=1, keepdim=True).float()
            best_orientTensor = maxResTensor * math.pi / self.numKernels

            orient_diff = torch.minimum(torch.abs(best_orientTensor - orient),
                               torch.minimum(torch.abs(best_orientTensor - orient - math.pi),
                                          torch.abs(best_orientTensor - orient + math.pi)))

            resp_diff = resTensor - max_resp
            variance = torch.sum(orient_diff * resp_diff * resp_diff, dim=1, keepdim=True)
            variance = variance ** (1 / 2)
            
            orient_data = torch.where(variance > variance_data, best_orientTensor, orient_data)
            max_resp_data = torch.where(variance > variance_data, max_resp, max_resp_data)
            variance_data = torch.where(variance > variance_data, variance, variance_data)

            max_all_resp = torch.max(max_resp_data)
            max_all_var = torch.max(variance_data)
            
            if max_all_resp > 0:
                max_resp_data = max_resp_data / max_all_resp
            if max_all_var > 0:
                variance_data = variance_data / max_all_var

            confidenceTensor = (variance_data - self.clamp_confidence_low) / (
                            self.clamp_confidence_high - self.clamp_confidence_low)
            confidenceTensor = confidenceTensor.clamp(0, 1)

            del resTensor, orient, max_resp, maxResTensor, best_orientTensor, orient_diff, resp_diff, variance
            torch.cuda.empty_cache()

            return confidenceTensor, variance_data, orient_data
            
        def forward(self, image, label, iter=1, threshold=0.0):
            H, W = image.size()[2:4]
            variance_data = torch.ones(1, 1, H, W).to(self.device) * 0
            orient_data = torch.ones(1, 1, H, W).to(self.device) * 0
            max_resp_data = torch.ones(1, 1, H, W).to(self.device) * 0

            for i in range(iter):
                confidenceTensor, variance_data, orient_data = self.filter(
                    image, label, threshold, variance_data, orient_data, max_resp_data
                )
                image = confidenceTensor
            
            confidenceTensor[confidenceTensor < threshold] = 0
            best_orientTensor = orient_data
            
            orientTwoChannel = torch.cat([torch.sin(best_orientTensor), torch.cos(best_orientTensor)], dim=1)
            return orientTwoChannel, best_orientTensor, confidenceTensor


# ============================================================================
# RESNET ENCODER
# ============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet encoder that extracts multi-scale features for UNet skip connections.
    Uses pre-trained ResNet backbone and returns intermediate features.
    """
    def __init__(self, backbone='resnet101', pretrained=True, input_channels=3):
        super(ResNetEncoder, self).__init__()
        
        # Load pre-trained ResNet
        if backbone == 'resnet18':
            self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_name = backbone
        self.input_channels = input_channels
        
        # Remove the unused FC layer
        self.resnet.fc = nn.Identity()
        
        # Handle variable input channels
        if input_channels != 3:
            original_conv1 = self.resnet.conv1
            
            new_conv1 = nn.Conv2d(
                input_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            with torch.no_grad():
                if input_channels > 3:
                    new_conv1.weight[:, :3, :, :] = original_conv1.weight
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                    new_conv1.weight[:, 3:, :, :] *= 0.1
                elif input_channels < 3:
                    new_conv1.weight = nn.Parameter(original_conv1.weight[:, :input_channels, :, :])
                
                if original_conv1.bias is not None:
                    new_conv1.bias = nn.Parameter(original_conv1.bias)
            
            self.resnet.conv1 = new_conv1
            
        # Extract ResNet layers
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        print(f"ResNet encoder initialized: {backbone}, features: {self.feature_channels}, input_ch: {input_channels}")
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)  # /2
        features.append(x)
        
        x = self.maxpool(x)  # /4
        
        x = self.layer1(x)  # /4
        features.append(x)
        
        x = self.layer2(x)  # /8
        features.append(x)
        
        x = self.layer3(x)  # /16
        features.append(x)
        
        x = self.layer4(x)  # /32
        features.append(x)
        
        return features


# ============================================================================
# DECODER BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class AttentionBlock(nn.Module):
    """Attention gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================================
# MOTION MODULES
# ============================================================================

class OpticalFlowEncoder(nn.Module):
    """Encode optical flow into feature maps."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, flow):
        return self.encoder(flow)


class MotionSegmentationHead(nn.Module):
    """Segment regions by motion characteristics."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features):
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return F.softmax(x, dim=1)


class TemporalFusionModule(nn.Module):
    """Fuse features from multiple frames for temporal consistency."""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, current_features, prev_features):
        if prev_features is None:
            return current_features
        
        if current_features.shape != prev_features.shape:
            prev_features = F.interpolate(
                prev_features, 
                size=current_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        concat = torch.cat([current_features, prev_features], dim=1)
        attention = self.attention(concat)
        
        fused = current_features * attention + prev_features * (1 - attention)
        return fused


# ============================================================================
# MAIN MODEL: UNet with ResNet Backbone
# ============================================================================

class MotionAwareUNet(nn.Module):
    """
    UNet with ResNet101 backbone for motion-aware cage segmentation.
    
    Architecture:
    - Encoder: ResNet101 (ImageNet pretrained)
    - Decoder: Custom decoder with skip connections
    - Optional: Gabor filter preprocessing
    - Optional: Motion feature integration
    - Optional: Attention gates
    """
    
    def __init__(
        self, 
        in_channels=3, 
        out_channels=1, 
        backbone='resnet101',
        decoder_channels=[1024, 512, 256, 128, 64],
        use_gabor=True,
        use_attention=True,
        use_motion=False,
        pretrained=True
    ):
        super().__init__()
        self.use_gabor = use_gabor
        self.use_attention = use_attention
        self.use_motion = use_motion
        self.backbone_name = backbone
        
        # Calculate total input channels
        total_input_channels = in_channels
        
        # Gabor orientation filter - adds 2 channels (sin/cos)
        if self.use_gabor:
            self.gabor_filter = calOrientationGabor(channel_in=1, channel_out=1)
            total_input_channels += 2
        
        # Motion modules
        if use_motion:
            self.flow_encoder = OpticalFlowEncoder(out_channels=32)
            self.motion_head = MotionSegmentationHead(in_channels + 32)
            self.temporal_fusion = TemporalFusionModule(decoder_channels[0])
            total_input_channels += 32
        
        # ResNet Encoder
        self.encoder = ResNetEncoder(
            backbone=backbone, 
            pretrained=pretrained, 
            input_channels=total_input_channels
        )
        
        encoder_channels = self.encoder.feature_channels
        
        # Center block
        self.center = DoubleConv(encoder_channels[-1], decoder_channels[0])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList() if use_attention else None
        
        for i in range(len(decoder_channels)):
            if i == 0:
                in_ch = decoder_channels[0]
                skip_ch = encoder_channels[-2]
                out_ch = decoder_channels[0]
            else:
                in_ch = decoder_channels[i-1]
                if len(encoder_channels) - 2 - i >= 0:
                    skip_ch = encoder_channels[len(encoder_channels) - 2 - i]
                else:
                    skip_ch = 0
                out_ch = decoder_channels[i]
            
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            
            if use_attention and skip_ch > 0:
                self.attention_blocks.append(AttentionBlock(in_ch, skip_ch, skip_ch // 2))
            elif use_attention:
                self.attention_blocks.append(None)
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        
        print(f"\nMotionAwareUNet initialized:")
        print(f"  - Backbone: {backbone}")
        print(f"  - Encoder channels: {encoder_channels}")
        print(f"  - Decoder channels: {decoder_channels}")
        print(f"  - Using Gabor: {use_gabor}")
        print(f"  - Using Attention: {use_attention}")
        print(f"  - Using Motion: {use_motion}")
        print(f"  - Total input channels: {total_input_channels}")

    def forward(self, x, flow=None, prev_features=None):
        """
        Args:
            x: Input image [B, 3, H, W]
            flow: Optional optical flow [B, 2, H, W]
            prev_features: Optional features from previous frame
            
        Returns:
            out: Segmentation mask [B, 1, H, W]
            motion_seg: Optional motion segmentation
            bottleneck_features: Features for temporal fusion
        """
        original_size = x.shape[2:]
        motion_seg = None
        
        # Motion encoding
        if self.use_motion and flow is not None:
            flow_features = self.flow_encoder(flow)
            if flow_features.shape[2:] != x.shape[2:]:
                flow_features = F.interpolate(flow_features, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            motion_input = torch.cat([x, flow_features], dim=1)
            motion_seg = self.motion_head(motion_input)
            x = motion_input
        elif self.use_motion:
            B, C, H, W = x.shape
            flow_features = torch.zeros(B, 32, H, W, device=x.device)
            x = torch.cat([x, flow_features], dim=1)
        
        # Gabor features
        if self.use_gabor:
            gray = torch.mean(x[:, :3, :, :], dim=1, keepdim=True)
            dummy_label = torch.ones_like(gray)
            
            # Move gabor filter to same device
            if hasattr(self, 'gabor_filter'):
                self.gabor_filter.device = x.device
            
            orientTwoChannel, best_ori, confidence = self.gabor_filter(gray, dummy_label)
            
            confidence_threshold = 0.2
            bconf = best_ori * (confidence > confidence_threshold)
            bconfTwoChannel = torch.cat([torch.sin(bconf), torch.cos(bconf)], dim=1)
            x = torch.cat([x, bconfTwoChannel], dim=1)
            
            del gray, dummy_label, orientTwoChannel, best_ori, confidence, bconf, bconfTwoChannel
            torch.cuda.empty_cache()
        
        # Encoder forward pass
        encoder_features = self.encoder(x)
        
        # Center processing
        center = self.center(encoder_features[-1])
        
        # Temporal fusion at bottleneck
        if self.use_motion and prev_features is not None:
            center = self.temporal_fusion(center, prev_features)
        
        bottleneck_features = center.clone()
        
        # Decoder forward pass
        x = center
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx >= 0:
                skip_feature = encoder_features[skip_idx]
                
                # Apply attention if available
                if self.use_attention and self.attention_blocks[i] is not None:
                    skip_feature = self.attention_blocks[i](x, skip_feature)
            else:
                skip_feature = None
            
            x = decoder_block(x, skip_feature)
        
        # Final resize to original size
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        
        # Segmentation output (raw logits - sigmoid applied in loss for autocast compatibility)
        out = self.segmentation_head(x)
        
        return out, motion_seg, bottleneck_features


class MotionAwareUncageNet(nn.Module):
    """
    Complete motion-aware cage removal network with ResNet backbone.
    Wraps MotionAwareUNet with additional utilities.
    """
    
    def __init__(
        self, 
        backbone='resnet101',
        use_gabor=True,
        use_motion=True, 
        use_attention=True,
        pretrained=True
    ):
        super().__init__()
        self.unet = MotionAwareUNet(
            in_channels=3,
            out_channels=1,
            backbone=backbone,
            decoder_channels=[1024, 512, 256, 128, 64],
            use_gabor=use_gabor,
            use_attention=use_attention,
            use_motion=use_motion,
            pretrained=pretrained
        )
        self.use_motion = use_motion
        self.prev_features = None
    
    def forward(self, x, flow=None):
        out, motion_seg, features = self.unet(x, flow, self.prev_features)
        
        if self.training:
            self.prev_features = features.detach()
        
        return out, motion_seg
    
    def reset_temporal(self):
        self.prev_features = None


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class MotionAwareLoss(nn.Module):
    """Combined loss for motion-aware cage segmentation."""
    
    def __init__(
        self, 
        bce_weight=0.5, 
        dice_weight=0.5, 
        motion_weight=0.0,
        seg_weight=1.0,
        smooth_l1=False
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.motion_weight = motion_weight
        self.seg_weight = seg_weight
        self.smooth_l1 = smooth_l1
        
        # Use BCEWithLogitsLoss for autocast compatibility (numerically stable)
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    def compute_metrics(self, pred, target, threshold=0.5):
        """Compute comprehensive segmentation metrics."""
        pred_binary = (pred > threshold).float()
        
        # Flatten for computation
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives, True Negatives
        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()
        TN = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        # Metrics
        smooth = 1e-6
        
        # IoU (Intersection over Union)
        iou = (TP + smooth) / (TP + FP + FN + smooth)
        
        # Dice Score (F1 for segmentation)
        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        
        # Precision
        precision = (TP + smooth) / (TP + FP + smooth)
        
        # Recall (Sensitivity)
        recall = (TP + smooth) / (TP + FN + smooth)
        
        # Specificity
        specificity = (TN + smooth) / (TN + FP + smooth)
        
        # Accuracy
        accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
        
        return {
            'iou': iou.item(),
            'dice_score': dice.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
            'accuracy': accuracy.item()
        }
    
    def forward(self, pred_logits, target, motion_seg=None, prev_pred=None):
        """Forward pass - pred_logits are raw logits (before sigmoid)."""
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.shape != pred_logits.shape:
            target = F.interpolate(target.float(), size=pred_logits.shape[2:], mode='nearest')
        
        # BCE loss on logits (autocast safe)
        bce_loss = self.bce(pred_logits, target)
        
        # Apply sigmoid for dice loss and metrics
        pred = torch.sigmoid(pred_logits)
        dice_loss = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # Motion loss (uses probabilities)
        motion_loss_val = 0.0
        if motion_seg is not None and self.motion_weight > 0:
            static_score = motion_seg[:, 0:1, :, :]
            if static_score.shape != pred.shape:
                static_score = F.interpolate(static_score, size=pred.shape[2:], mode='bilinear', align_corners=True)
            
            motion_loss = torch.mean(pred * (1 - static_score))
            motion_loss_val = motion_loss.item()
            total_loss += self.motion_weight * motion_loss
        
        # Temporal loss (uses probabilities)
        temporal_loss_val = 0.0
        if prev_pred is not None:
            if prev_pred.shape != pred.shape:
                prev_pred = F.interpolate(prev_pred, size=pred.shape[2:], mode='bilinear', align_corners=True)
            temporal_loss = F.l1_loss(pred, prev_pred)
            temporal_loss_val = temporal_loss.item()
            total_loss += 0.1 * temporal_loss
        
        # Compute segmentation metrics (uses probabilities)
        metrics = self.compute_metrics(pred, target)
        
        return total_loss, {
            'bce': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'motion_loss': motion_loss_val,
            'temporal_loss': temporal_loss_val,
            'total': total_loss.item(),
            # Segmentation metrics
            'iou': metrics['iou'],
            'dice_score': metrics['dice_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy']
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_motion_aware_uncagenet(
    in_channels=3, 
    out_channels=1, 
    base_channels=64,
    backbone='resnet101',
    use_gabor=True,
    use_motion=True, 
    use_attention=True,
    pretrained=True,
    pretrained_path=None
):
    """
    Create a MotionAwareUncageNet model with ResNet backbone.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary mask)
        base_channels: Base number of channels (not used, kept for compatibility)
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        use_gabor: Whether to use Gabor filter preprocessing
        use_motion: Whether to use motion features
        use_attention: Whether to use attention gates
        pretrained: Whether to use ImageNet pretrained weights
        pretrained_path: Path to pretrained model weights
        
    Returns:
        model: MotionAwareUncageNet instance
    """
    model = MotionAwareUncageNet(
        backbone=backbone,
        use_gabor=use_gabor,
        use_motion=use_motion, 
        use_attention=use_attention,
        pretrained=pretrained
    )
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
    
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_optical_flow_opencv(frame1, frame2):
    """Compute optical flow using OpenCV's Farneback method."""
    import cv2
    print("Computing optical flow using OpenCV Farneback method...")
    
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
    
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow


def flow_to_tensor(flow, device='cpu'):
    """Convert numpy optical flow to tensor."""
    flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float().unsqueeze(0)
    return flow_tensor.to(device)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing MotionAwareUncageNet with ResNet101 Backbone")
    print("="*70)
    
    # Create model
    model = create_motion_aware_uncagenet(
        backbone='resnet101',
        use_gabor=False,  # Disable for quick test
        use_motion=True, 
        use_attention=True,
        pretrained=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = torch.randn(2, 3, 256, 144).to(device)
    flow = torch.randn(2, 2, 256, 144).to(device)
    
    with torch.no_grad():
        out, motion_seg = model(x, flow)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Output shape: {out.shape}")
    if motion_seg is not None:
        print(f"Motion seg shape: {motion_seg.shape}")
    
    # Test loss
    loss_fn = MotionAwareLoss()
    target = torch.rand(2, 1, 256, 144).to(device)
    loss, components = loss_fn(out, target, motion_seg)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Components: {components}")
    
    print("\n" + "="*70)
    print("✓ All tests passed! ResNet101 backbone ready for training.")
    print("="*70)
"""

tensorboard --logdir motion_aware_resnet_outputs/tensorboard --port 6006 --bind_all
cd /mnt/zone/B/Sayak && source ~/anaconda3/etc/profile.d/conda.sh && conda activate deepfill && CUDA_VISIBLE_DEVICES=1 python train_motion_aware.py   --data_root /mnt/zone/B/Sayak/dataset/organized   --batch_size 24  --epochs 100   --backbone resnet101   --use_gabor   --use_attention   --use_motion
"""