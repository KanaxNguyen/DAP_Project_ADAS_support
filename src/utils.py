"""
src/utils.py – Hàm Tiện Ích
==============================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Cung cấp các hàm tiện ích:
    - xywh2xyxy, xyxy2xywh: Chuyển đổi định dạng bounding box.
    - box_iou: Tính IoU giữa hai tập bounding box.
    - non_max_suppression (NMS): Loại bỏ bounding box trùng lặp.
      Lấy cảm hứng từ thuật toán NMS chuẩn trong cộng đồng detection.

Lưu ý: Code tự viết, không dùng thư viện torchvision.ops.nms.
"""
import torch
import numpy as np

def xywh2xyxy(x):
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    Can handle both PyTorch Tensors and NumPy arrays.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]      # width
    y[..., 3] = x[..., 3] - x[..., 1]      # height
    return y

def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes (x1, y1, x2, y2).
    box1: [N, 4]
    box2: [M, 4]
    Returns: [N, M] tensor containing the pairwise IoU values.
    """
    # area of boxes
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # union
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-16)

def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45):
    """
    Runs Non-Maximum Suppression (NMS) on inference results.
    predictions: tensor of shape [B, S, S, 5 + C]
                 where the 5 is [conf, x_cell, y_cell, w_img, h_img] and C is class scores.
    Returns:
        List of detections, each of shape [N, 6] -> [x1, y1, x2, y2, confidence, class_id]
    """
    device = predictions.device
    batch_size = predictions.shape[0]
    grid_size = predictions.shape[1]
    
    output = [torch.zeros((0, 6), device=device) for _ in range(batch_size)]
    
    # Pre-calculate cell offsets
    # grid_x, grid_y are shape [S, S]
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_size, device=device), torch.arange(grid_size, device=device), indexing='ij')
    
    for image_idx in range(batch_size):
        pred = predictions[image_idx] # [S, S, 5 + C]
        
        # Extract components
        obj_conf = pred[..., 0] # [S, S]
        xy = pred[..., 1:3]     # [S, S, 2] - relative to cell (0 to 1)
        wh = pred[..., 3:5]     # [S, S, 2] - relative to image (0 to 1)
        class_scores = pred[..., 5:] # [S, S, C]
        
        # Get class IDs and max class score
        class_confs, class_ids = torch.max(class_scores, dim=-1) # [S, S]
        
        # Combined confidence
        scores = obj_conf * class_confs # [S, S]
        
        # Filter by confidence threshold
        mask = scores > conf_thres # [S, S]
        if not mask.any():
            continue
            
        # Extract valid predictions
        valid_scores = scores[mask]
        valid_class_ids = class_ids[mask]
        
        # Decode bounding box coordinates to absolute/image relative [0, 1] xywh
        # x_center_img = (x_cell + cell_x_index) / S
        # y_center_img = (y_cell + cell_y_index) / S
        cell_x = grid_x[mask].float()
        cell_y = grid_y[mask].float()
        
        x_center = (xy[mask][:, 0] + cell_x) / grid_size
        y_center = (xy[mask][:, 1] + cell_y) / grid_size
        w = wh[mask][:, 0]
        h = wh[mask][:, 1]
        
        # Format as [x_center, y_center, w, h]
        boxes_xywh = torch.stack([x_center, y_center, w, h], dim=1)
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        
        # Keep inside image bounds [0, 1]
        boxes_xyxy[:, 0:4] = torch.clamp(boxes_xyxy[:, 0:4], 0.0, 1.0)
        
        # Form matrix [x1, y1, x2, y2, score, class_id]
        detections = torch.cat([boxes_xyxy, valid_scores[:, None], valid_class_ids[:, None].float()], dim=1)
        
        # Sort by confidence descending
        indices = torch.argsort(detections[:, 4], descending=True)
        detections = detections[indices]
        
        # PyTorch NMS
        keep_boxes = []
        while detections.shape[0] > 0:
            best_box = detections[0]
            keep_boxes.append(best_box)
            if detections.shape[0] == 1:
                break
            # Calculate IoU of the best box with all others
            ious = box_iou(best_box[None, 0:4], detections[1:, 0:4])[0]
            # Keep only boxes with IoU less than iou_thres
            detections = detections[1:][ious < iou_thres]
            
        if len(keep_boxes) > 0:
            output[image_idx] = torch.stack(keep_boxes, dim=0)
            
    return output
