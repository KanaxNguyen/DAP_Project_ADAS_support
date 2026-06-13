"""
src/model.py – Kiến Trúc Mạng Đa Nhiệm (Multi-Task Network)
==============================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Định nghĩa ADASMultiTaskNetwork:
    - Backbone: ResNet18/34 pretrained (He et al., CVPR 2016) via torchvision
    - Neck: Feature Pyramid Network (FPN) (Lin et al., CVPR 2017)
    - Detection Head: YOLO-style grid prediction (lấy cảm hứng từ Redmon et al., CVPR 2016)
    - Segmentation Head: Multi-scale fusion decoder

Nguồn tham khảo kiến trúc:
    [1] He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
        ResNet backbone – pretrained weights tải từ torchvision.
    [2] Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
        FPN neck – top-down lateral connections.
    [3] Redmon et al., "You Only Look Once" series.
        Phong cách đầu phát hiện theo lưới ô vuông (grid-based detection head).

Lưu ý: Đây là nguyên mẫu học thuật, không phải sản phẩm thương mại.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ADASMultiTaskNetwork(nn.Module):
    def __init__(self, backbone_name="resnet18", grid_size=40, num_classes_det=7, num_classes_seg=4, pretrained=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes_det = num_classes_det
        self.num_classes_seg = num_classes_seg
        
        # Load backbone
        if backbone_name == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT if hasattr(models, 'ResNet18_Weights') else None
                self.backbone = models.resnet18(weights=weights) if weights else models.resnet18(pretrained=True)
            else:
                self.backbone = models.resnet18(pretrained=False)
            c2_channels, c3_channels, c4_channels = 128, 256, 512
        elif backbone_name == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT if hasattr(models, 'ResNet34_Weights') else None
                self.backbone = models.resnet34(weights=weights) if weights else models.resnet34(pretrained=True)
            else:
                self.backbone = models.resnet34(pretrained=False)
            c2_channels, c3_channels, c4_channels = 128, 256, 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Feature extraction layers from ResNet
        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        ) # Stride 4, output 64 channels
        
        self.layer1 = self.backbone.layer1 # Stride 4, output 64 channels
        self.layer2 = self.backbone.layer2 # Stride 8, output 128 channels (c2)
        self.layer3 = self.backbone.layer3 # Stride 16, output 256 channels (c3)
        self.layer4 = self.backbone.layer4 # Stride 32, output 512 channels (c4)
        
        # FPN Projections to common dimension (128 channels)
        fpn_channels = 128
        self.proj_c4 = nn.Conv2d(c4_channels, fpn_channels, kernel_size=1)
        self.proj_c3 = nn.Conv2d(c3_channels, fpn_channels, kernel_size=1)
        self.proj_c2 = nn.Conv2d(c2_channels, fpn_channels, kernel_size=1)
        
        # FPN Smoothing Convolutions
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        
        # Detection Head
        # Takes p3 (stride 16) or interpolated features and predicts grid cell outputs.
        # Output channel size: 5 + C (conf, x_cell, y_cell, w_img, h_img, class_scores)
        self.det_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, 5 + self.num_classes_det, kernel_size=1)
        )
        
        # Segmentation Head
        # Concatenates upsampled p2, p3, p4 and projects to num_classes_seg
        self.seg_conv1 = nn.Conv2d(fpn_channels * 3, fpn_channels, kernel_size=3, padding=1)
        self.seg_bn1 = nn.BatchNorm2d(fpn_channels)
        self.seg_conv2 = nn.Conv2d(fpn_channels, 64, kernel_size=3, padding=1)
        self.seg_bn2 = nn.BatchNorm2d(64)
        self.seg_head = nn.Conv2d(64, self.num_classes_seg, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [B, 3, H, W] (typically 640x640)
        h_in, w_in = x.shape[2], x.shape[3]
        
        # Shared Encoder Forward Pass
        x = self.stem(x)   # Stride 4
        x = self.layer1(x) # Stride 4
        c2 = self.layer2(x) # Stride 8, shape: [B, 128, H/8, W/8]
        c3 = self.layer3(c2) # Stride 16, shape: [B, 256, H/16, W/16]
        c4 = self.layer4(c3) # Stride 32, shape: [B, 512, H/32, W/32]
        
        # FPN Neck Forward Pass
        p4 = self.proj_c4(c4)
        p3 = self.proj_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.proj_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        
        # Smooth features
        p4 = self.smooth_p4(p4) # Stride 32
        p3 = self.smooth_p3(p3) # Stride 16
        p2 = self.smooth_p2(p2) # Stride 8
        
        # 1. Detection Head
        # Rescale p3 features to match target grid_size S x S
        det_feats = F.interpolate(p3, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        det_out = self.det_head(det_feats) # [B, 5 + C, S, S]
        det_out = det_out.permute(0, 2, 3, 1) # [B, S, S, 5 + C]
        
        # 2. Segmentation Head
        # Upsample p3 and p4 to match p2 resolution (stride 8)
        p3_up = F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=p2.shape[2:], mode="bilinear", align_corners=False)
        
        # Fused multi-scale representations
        seg_feats = torch.cat([p2, p3_up, p4_up], dim=1) # [B, fpn_channels * 3, H/8, W/8]
        
        # Apply segmentation layers
        seg_out = F.relu(self.seg_bn1(self.seg_conv1(seg_feats)))
        seg_out = F.relu(self.seg_bn2(self.seg_conv2(seg_out)))
        seg_out = self.seg_head(seg_out) # [B, num_classes_seg, H/8, W/8]
        
        # Upsample to match original input image resolution
        seg_out = F.interpolate(seg_out, size=(h_in, w_in), mode="bilinear", align_corners=False)
        
        return det_out, seg_out
