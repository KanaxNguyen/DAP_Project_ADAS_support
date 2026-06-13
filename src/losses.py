"""
src/losses.py – Tổ Hợp Hàm Mất Mát Đa Nhiệm
==============================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Triển khai các hàm mất mát tự viết, lấy cảm hứng từ các công trình:

Nguồn tham khảo:
    [1] Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), ICCV 2017.
        FocalLoss – giảm đóng góp của mẫu dễ phân loại, tập trung vào mẫu khó.
    [2] Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression",
        AAAI 2020. CIoU Loss – tính đồng thời IoU, khoảng cách tâm và tỷ lệ khích thước.
    [3] Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical
        Image Segmentation", 3DV 2016. Dice Loss – phù hợp với lớp không cân bằng (imbalanced).

Lưu ý: Code tự viết hoàn toàn (from scratch), không cóp từ repository gốc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import xywh2xyxy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits (before sigmoid)
        targets: binary targets (0 or 1)
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_ciou_loss(pred_boxes, target_boxes):
    """
    Computes CIoU loss between pred_boxes and target_boxes.
    Both boxes are of shape [N, 4] in (x1, y1, x2, y2) format.
    """
    # Coordinates of intersection box
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Area of prediction and target boxes
    w1, h1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0), (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    w2, h2 = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0), (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)
    union_area = (w1 * h1) + (w2 * h2) - inter_area + 1e-16
    
    iou = inter_area / union_area
    
    # Coords of enclosing box
    cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Diagonal length of enclosing box
    c_diag_sq = (cx2 - cx1)**2 + (cy2 - cy1)**2 + 1e-16
    
    # Centers distance
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    rho_sq = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
    
    # Aspect ratio consistency
    atan_target = torch.atan(w2 / (h2 + 1e-16))
    atan_pred = torch.atan(w1 / (h1 + 1e-16))
    v = (4 / (np.pi ** 2)) * torch.pow(atan_target - atan_pred, 2)
    
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-16)
        
    ciou = iou - (rho_sq / c_diag_sq) - alpha * v
    return (1.0 - ciou).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] (class indices)
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # One-hot target
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 0.0
        for c in range(num_classes):
            p_c = probs[:, c]
            t_c = targets_one_hot[:, c]
            
            intersection = (p_c * t_c).sum(dim=(1, 2))
            union = p_c.sum(dim=(1, 2)) + t_c.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice)
            
        return dice_loss.mean() / num_classes

class ADASMultiTaskLoss(nn.Module):
    def __init__(self, grid_size=40, num_classes_det=7, num_classes_seg=4, lambda_det=1.0, lambda_seg=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes_det = num_classes_det
        self.num_classes_seg = num_classes_seg
        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        self.dice_loss = DiceLoss()
        
    def forward(self, pred_det, pred_seg, target_det, target_seg):
        """
        pred_det: [B, S, S, 5 + C] (logits for conf, raw outputs for xywh, logits for classes)
        pred_seg: [B, num_classes_seg, H, W] (raw logits)
        target_det: [B, S, S, 5 + C] (targets)
        target_seg: [B, H, W] (segmentation class indices)
        """
        device = pred_det.device
        
        # 1. SEGMENTATION LOSS
        ce_seg = F.cross_entropy(pred_seg, target_seg)
        dice_seg = self.dice_loss(pred_seg, target_seg)
        seg_loss = ce_seg + dice_seg
        
        # 2. DETECTION LOSS
        # Targets mapping
        target_obj = target_det[..., 0] # [B, S, S]
        pred_obj_logits = pred_det[..., 0] # [B, S, S]
        
        # Objectness loss using Focal Loss
        loss_obj = self.focal_loss(pred_obj_logits, target_obj)
        
        # Mask for cells containing objects
        pos_mask = (target_obj == 1)
        
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        num_pos = pos_mask.sum().item()
        
        if num_pos > 0:
            # Box Regression Loss (CIoU)
            # Pred components
            xy_pred = torch.sigmoid(pred_det[pos_mask][:, 1:3]) # x, y relative to cell (0 to 1)
            wh_pred = torch.exp(pred_det[pos_mask][:, 3:5]) # w, h relative to image (exponential anchor-free activation)
            
            # Target components
            xy_target = target_det[pos_mask][:, 1:3]
            wh_target = target_det[pos_mask][:, 3:5]
            
            # Reconstruct cell indices to decode coordinates
            grid_y, grid_x = torch.meshgrid(torch.arange(self.grid_size, device=device), torch.arange(self.grid_size, device=device), indexing='ij')
            grid_x = grid_x.unsqueeze(0).expand_as(pos_mask)
            grid_y = grid_y.unsqueeze(0).expand_as(pos_mask)
            cell_x = grid_x[pos_mask].float()
            cell_y = grid_y[pos_mask].float()
            
            # Decode to image relative [0, 1] xywh
            pred_x_img = (xy_pred[:, 0] + cell_x) / self.grid_size
            pred_y_img = (xy_pred[:, 1] + cell_y) / self.grid_size
            
            target_x_img = (xy_target[:, 0] + cell_x) / self.grid_size
            target_y_img = (xy_target[:, 1] + cell_y) / self.grid_size
            
            pred_boxes_xywh = torch.stack([pred_x_img, pred_y_img, wh_pred[:, 0], wh_pred[:, 1]], dim=1)
            target_boxes_xywh = torch.stack([target_x_img, target_y_img, wh_target[:, 0], wh_target[:, 1]], dim=1)
            
            # Convert to xyxy
            pred_boxes_xyxy = xywh2xyxy(pred_boxes_xywh)
            target_boxes_xyxy = xywh2xyxy(target_boxes_xywh)
            
            loss_box = compute_ciou_loss(pred_boxes_xyxy, target_boxes_xyxy)
            
            # Classification Loss (Cross Entropy)
            pred_cls_logits = pred_det[pos_mask][:, 5:] # [num_pos, C]
            target_cls = torch.argmax(target_det[pos_mask][:, 5:], dim=-1) # [num_pos]
            loss_cls = F.cross_entropy(pred_cls_logits, target_cls)
            
        det_loss = loss_obj + 2.0 * loss_box + loss_cls
        
        # Combine
        total_loss = self.lambda_det * det_loss + self.lambda_seg * seg_loss
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_det": det_loss.item(),
            "loss_obj": loss_obj.item(),
            "loss_box": loss_box.item(),
            "loss_cls": loss_cls.item(),
            "loss_seg": seg_loss.item(),
            "loss_ce_seg": ce_seg.item(),
            "loss_dice_seg": dice_seg.item()
        }
