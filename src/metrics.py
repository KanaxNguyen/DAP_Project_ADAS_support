"""
src/metrics.py – Đánh Giá Hiệu Năng: mAP@0.5 và mIoU
=====================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    - evaluate_detection(): Tính Precision, Recall, AP per class và mAP@0.5.
      Phương pháp tichân của diện tích dưới đường cong PR (PR-AUC).
    - SegmentationMetricTracker: Tính mIoU và pixel accuracy
      dựa trên Confusion Matrix.

Lưu ý: Cách tính AP là phương pháp chuẩn trong cộng đồng computer vision
    (tương tự Pascal VOC đánh giá 2010+). Code tự viết.
"""
import torch
import numpy as np
from src.utils import xywh2xyxy

def compute_ap(recall, precision):
    """
    Computes Area Under PR Curve (AP) using the 101-point interpolation or area-based method.
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Integrate area under curve
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_detection(pred_list, target_list, num_classes=7, iou_threshold=0.5):
    """
    Evaluates detection metrics (Precision, Recall, mAP@0.5).
    pred_list: list of np.arrays of shape [N, 6] -> [x1, y1, x2, y2, conf, class_id] (normalized [0, 1])
    target_list: list of np.arrays of shape [M, 5] -> [class_id, x, y, w, h] (normalized [0, 1])
    """
    # Store TPs, FPs, and scores for each class
    class_stats = {c: {"tp": [], "fp": [], "scores": [], "total_gt": 0} for c in range(num_classes)}
    
    # First count total GT per class
    for targets in target_list:
        if len(targets) == 0:
            continue
        for class_id in targets[:, 0]:
            if int(class_id) in class_stats:
                class_stats[int(class_id)]["total_gt"] += 1
                
    # Helper pairwise IoU
    def calculate_iou_np(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / (unionArea + 1e-16)

    # Process predictions image by image
    for idx, (preds, targets) in enumerate(zip(pred_list, target_list)):
        if len(preds) == 0:
            continue
            
        # Convert target boxes format to [x1, y1, x2, y2, class_id]
        if len(targets) > 0:
            target_boxes_xyxy = xywh2xyxy(targets[:, 1:5])
            gt_classes = targets[:, 0].astype(int)
            gt_matched = np.zeros(len(targets), dtype=bool)
        else:
            target_boxes_xyxy = np.zeros((0, 4))
            gt_classes = np.array([], dtype=int)
            gt_matched = np.array([], dtype=bool)
            
        # Sort predictions by confidence score descending
        preds = preds[np.argsort(preds[:, 4])[::-1]]
        
        for pred in preds:
            pred_box = pred[:4]
            pred_score = pred[4]
            pred_class = int(pred[5])
            
            best_iou = -1.0
            best_gt_idx = -1
            
            # Find matching ground truth of the same class
            for gt_idx, gt_box in enumerate(target_boxes_xyxy):
                if gt_classes[gt_idx] == pred_class:
                    iou = calculate_iou_np(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
            # Determine if TP or FP
            if best_iou >= iou_threshold and best_gt_idx != -1:
                if not gt_matched[best_gt_idx]:
                    class_stats[pred_class]["tp"].append(1)
                    class_stats[pred_class]["fp"].append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    # GT already matched by a higher-confidence prediction
                    class_stats[pred_class]["tp"].append(0)
                    class_stats[pred_class]["fp"].append(1)
            else:
                class_stats[pred_class]["tp"].append(0)
                class_stats[pred_class]["fp"].append(1)
            class_stats[pred_class]["scores"].append(pred_score)
            
    # Calculate Precision, Recall, and AP per class
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for c in range(num_classes):
        stats = class_stats[c]
        tps = np.array(stats["tp"])
        fps = np.array(stats["fp"])
        scores = np.array(stats["scores"])
        n_gt = stats["total_gt"]
        total_gt += n_gt
        
        if len(tps) == 0:
            aps.append(0.0)
            continue
            
        # Sort by confidence score
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        fps = fps[sort_indices]
        
        # Cumulative sums
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        
        total_tp += np.sum(tps)
        total_fp += np.sum(fps)
        
        # Precision & Recall
        precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
        recalls = tp_cum / (n_gt + 1e-16)
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
    precision = total_tp / (total_tp + total_fp + 1e-16)
    recall = total_tp / (total_gt + 1e-16)
    mAP = np.mean(aps)
    
    return {
        "precision": precision,
        "recall": recall,
        "mAP": mAP,
        "ap_per_class": aps
    }

class SegmentationMetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        # Accumulate confusion matrix across evaluations
        # Row: Ground Truth, Column: Prediction
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def update(self, pred_mask, target_mask):
        """
        pred_mask: np.array of shape [B, H, W] or [H, W] (class IDs)
        target_mask: np.array of shape [B, H, W] or [H, W] (class IDs)
        """
        pred = pred_mask.flatten()
        target = target_mask.flatten()
        
        # Filter out invalid labels if any
        valid_mask = (target >= 0) & (target < self.num_classes)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute confusion matrix index
        # index = class * gt_class + pred_class
        # This is extremely fast using bincount!
        bins = self.num_classes * target + pred
        counts = np.bincount(bins, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))
        
    def get_results(self):
        """
        Computes IoU for each class and mIoU.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp
        
        union = tp + fn + fp
        iou = tp / (union + 1e-16)
        
        # Accuracy
        pixel_acc = np.sum(tp) / (np.sum(cm) + 1e-16)
        
        return {
            "mIoU": np.mean(iou),
            "iou_per_class": iou,
            "pixel_accuracy": pixel_acc
        }
