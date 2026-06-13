"""
src/sahi_inference.py – Sliced Inference kiểu SAHI
=====================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Triển khai phương pháp chia nhỏ ảnh (sliced inference) để cải thiện
    độ chính xác phát hiện vật thể nhỏ ở độ phân giải cao.

Nguồn tham khảo:
    Akyon et al., "Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection",
    ICIP 2022. (https://github.com/obss/sahi)
    Lưu ý: Đây là triển khai tự viết, không dùng thư viện SAHI gốc.
    Chỉ áp dụng ý tưởng chia ảnh + merge NMS, không cóp code.
"""
import torch
import numpy as np
import cv2
from src.utils import non_max_suppression, xyxy2xywh

def get_slices(image_height, image_width, slice_size=640, overlap_ratio=0.25):
    """
    Computes coordinates for overlapping slices.
    """
    step = int(slice_size * (1.0 - overlap_ratio))
    
    y_starts = []
    y = 0
    while y < image_height:
        if y + slice_size > image_height:
            y_starts.append(max(0, image_height - slice_size))
            break
        y_starts.append(y)
        y += step
        
    x_starts = []
    x = 0
    while x < image_width:
        if x + slice_size > image_width:
            x_starts.append(max(0, image_width - slice_size))
            break
        x_starts.append(x)
        x += step
        
    slice_coords = []
    for y_start in y_starts:
        for x_start in x_starts:
            slice_coords.append((x_start, y_start, x_start + slice_size, y_start + slice_size))
            
    return slice_coords

def sahi_sliced_inference(model, image, slice_size=640, overlap_ratio=0.25, conf_threshold=0.25, iou_threshold=0.45, device="cpu"):
    """
    Runs SAHI-style sliced inference on a high-resolution input image.
    image: numpy array of shape [H, W, 3] (RGB)
    Returns:
        detections: numpy array of shape [N, 6] -> [x1, y1, x2, y2, confidence, class_id] (mapped to original image)
        seg_mask: numpy array of shape [H, W] (class IDs)
    """
    h_orig, w_orig = image.shape[:2]
    
    # If image is already smaller than or equal to slice size, run normal inference
    if h_orig <= slice_size and w_orig <= slice_size:
        # Resize to slice_size x slice_size to fit model input exactly
        image_resized = cv2.resize(image, (slice_size, slice_size))
        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            det_out, seg_out = model(image_tensor)
            detections_list = non_max_suppression(det_out, conf_threshold, iou_threshold)
            detections = detections_list[0].cpu().numpy()
            
            # Map detections back to original scale
            if len(detections) > 0:
                detections[:, [0, 2]] *= (w_orig / slice_size)
                detections[:, [1, 3]] *= (h_orig / slice_size)
                
            seg_probs = torch.softmax(seg_out, dim=1)
            seg_mask = torch.argmax(seg_probs[0], dim=0).cpu().numpy().astype(np.uint8)
            seg_mask = cv2.resize(seg_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        return detections, seg_mask

    # Retrieve slice coordinates
    slice_coords = get_slices(h_orig, w_orig, slice_size, overlap_ratio)
    
    all_detections = []
    
    # Segmentation buffers
    num_classes_seg = model.num_classes_seg
    seg_accum = np.zeros((num_classes_seg, h_orig, w_orig), dtype=np.float32)
    seg_count = np.zeros((h_orig, w_orig), dtype=np.float32)
    
    for x1, y1, x2, y2 in slice_coords:
        slice_crop = image[y1:y2, x1:x2]
        
        # Preprocess crop
        crop_tensor = torch.from_numpy(slice_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            det_out, seg_out = model(crop_tensor)
            
            # 1. Decode detection bounding boxes
            # NMS outputs detections relative to the slice (scaled [0, 1])
            detections_list = non_max_suppression(det_out, conf_threshold, iou_threshold)
            slice_dets = detections_list[0].cpu().numpy()
            
            for det in slice_dets:
                # det: [x1_rel, y1_rel, x2_rel, y2_rel, conf, class_id]
                # Convert slice-relative coordinates [0, 1] to pixel coordinates [0, slice_size]
                x1_px = det[0] * slice_size
                y1_px = det[1] * slice_size
                x2_px = det[2] * slice_size
                y2_px = det[3] * slice_size
                
                # Shift to full image coordinates
                x1_full = x1_px + x1
                y1_full = y1_px + y1
                x2_full = x2_px + x1
                y2_full = y2_px + y1
                
                all_detections.append([x1_full, y1_full, x2_full, y2_full, det[4], det[5]])
                
            # 2. Accumulate Segmentation Masks
            # seg_out shape: [1, num_classes, slice_size, slice_size]
            seg_probs = torch.softmax(seg_out, dim=1).squeeze(0).cpu().numpy() # [num_classes, slice_size, slice_size]
            seg_accum[:, y1:y2, x1:x2] += seg_probs
            seg_count[y1:y2, x1:x2] += 1.0

    # 3. Merge detection bounding boxes using global NMS
    final_detections = np.zeros((0, 6), dtype=np.float32)
    if len(all_detections) > 0:
        all_detections = np.array(all_detections, dtype=np.float32)
        
        # Convert to tensors for box_iou and NMS operations
        boxes = torch.from_numpy(all_detections[:, :4])
        scores = torch.from_numpy(all_detections[:, 4])
        classes = torch.from_numpy(all_detections[:, 5])
        
        # Apply manual PyTorch NMS
        keep = []
        indices = torch.argsort(scores, descending=True)
        
        # Helper pairwise IoU
        def box_iou_numpy(b1, b2):
            area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
            area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
            lt = np.maximum(b1[:, None, :2], b2[:, :2])
            rb = np.minimum(b1[:, None, 2:], b2[:, 2:])
            wh = np.clip(rb - lt, a_min=0, a_max=None)
            inter = wh[:, :, 0] * wh[:, :, 1]
            union = area1[:, None] + area2 - inter
            return inter / (union + 1e-16)

        dets_np = all_detections[indices.numpy()]
        while dets_np.shape[0] > 0:
            best_det = dets_np[0]
            keep.append(best_det)
            if dets_np.shape[0] == 1:
                break
            ious = box_iou_numpy(best_det[None, :4], dets_np[1:, :4])[0]
            # Match only if class matches too, or suppress all close boxes. 
            # In SAHI, standard NMS is class-specific or class-agnostic. Let's do class-specific NMS.
            class_mask = (dets_np[1:, 5] == best_det[5])
            suppress_mask = (ious >= iou_threshold) & class_mask
            # Also suppress boxes with extremely high IoU of any class to avoid duplicates
            suppress_mask = suppress_mask | (ious >= 0.8)
            dets_np = dets_np[1:][~suppress_mask]
            
        final_detections = np.array(keep, dtype=np.float32)
        
        # Re-normalize to [0, 1] relative to the original image dimensions for consistency if needed,
        # but standard is to return absolute coordinates for visualization. 
        # Let's return absolute coordinates in pixels, since the viz / tracker can use pixels!
        # Wait, the tracker expects pixel coordinates or relative? Let's check:
        # In tracker.py, we run greedy matching with `calculate_iou(track.bbox, det[:4])`. 
        # IoU is coordinate-system independent, so absolute pixel coordinates are perfect!

    # 4. Finalize Segmentation Mask by averaging probabilities and taking argmax
    seg_count = np.clip(seg_count, a_min=1.0, a_max=None)
    seg_accum = seg_accum / seg_count[None, :, :]
    seg_mask = np.argmax(seg_accum, axis=0).astype(np.uint8)
    
    return final_detections, seg_mask
