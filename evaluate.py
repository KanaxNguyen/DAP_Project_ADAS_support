"""
evaluate.py – Script Đánh Giá Mô Hình ADAS
============================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Chạy vídu: python evaluate.py --config configs/config.yaml --weights checkpoints/best.pt
Điều phối:
    - Tính mAP@0.5 cho bài toán phát hiện biển báo.
    - Tính mIoU và pixel accuracy cho bài toán phân vùng.
"""
import os
import argparse
import yaml
import torch
import numpy as np
from src.dataset import get_dataloader
from src.model import ADASMultiTaskNetwork
from src.utils import non_max_suppression
from src.metrics import evaluate_detection, SegmentationMetricTracker

# BGR text styles for terminal print
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

CLASS_NAMES_DET = [
    "P127_40", "P127_50", "P127_60", "P127_80", "P127_90", 
    "R420_enter_residential", "R421_exit_residential"
]

CLASS_NAMES_SEG = [
    "background", "lane_marking", "hard_divider", "drivable_area"
]

def evaluate(config_path, weights_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['train']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Load Validation Dataloader
    val_loader = get_dataloader(
        img_dir=data_cfg['val_images'],
        label_dir=data_cfg['val_labels'],
        mask_dir=data_cfg['val_masks'],
        img_size=data_cfg['image_size'],
        grid_size=model_cfg['grid_size'],
        num_classes_det=model_cfg['num_classes_det'],
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Instantiate Model
    model = ADASMultiTaskNetwork(
        backbone_name=model_cfg['backbone'],
        grid_size=model_cfg['grid_size'],
        num_classes_det=model_cfg['num_classes_det'],
        num_classes_seg=model_cfg['num_classes_seg'],
        pretrained=False
    ).to(device)
    
    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}. Please train the model first.")
        
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from {weights_path} (trained for {checkpoint.get('epoch', 'N/A')} epochs)")
    
    model.eval()
    
    # Metrics Trackers
    seg_tracker = SegmentationMetricTracker(num_classes=model_cfg['num_classes_seg'])
    pred_boxes_all = []
    target_boxes_all = []
    
    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for images, masks, encoded_dets, raw_boxes in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            det_out, seg_out = model(images)
            
            # Segmentation updates
            seg_probs = torch.softmax(seg_out, dim=1)
            pred_seg_masks = torch.argmax(seg_probs, dim=1).cpu().numpy()
            target_seg_masks = masks.cpu().numpy()
            seg_tracker.update(pred_seg_masks, target_seg_masks)
            
            # Detection updates
            detections = non_max_suppression(det_out, conf_thres=0.1, iou_thres=0.45)
            for det, raw_box in zip(detections, raw_boxes):
                pred_boxes_all.append(det.cpu().numpy())
                target_boxes_all.append(raw_box.cpu().numpy())
                
    # 1. Compute Detection Results
    det_results = evaluate_detection(
        pred_boxes_all, target_boxes_all, num_classes=model_cfg['num_classes_det']
    )
    
    # 2. Compute Segmentation Results
    seg_results = seg_tracker.get_results()
    
    # Print clean report
    print("\n" + "="*50)
    print(f"{BLUE}             ADAS EVALUATION REPORT{RESET}")
    print("="*50)
    
    print(f"\n{GREEN}[+] OBJECT DETECTION METRICS:{RESET}")
    print(f"  - Precision: {det_results['precision']:.4f}")
    print(f"  - Recall:    {det_results['recall']:.4f}")
    print(f"  - mAP@0.5:   {det_results['mAP']:.4f}")
    print("\n  Class-wise Average Precision (AP@0.5):")
    for i, class_name in enumerate(CLASS_NAMES_DET):
        ap = det_results['ap_per_class'][i]
        print(f"    * {class_name:<25}: {ap:.4f}")
        
    print(f"\n{GREEN}[+] SEMANTIC SEGMENTATION METRICS:{RESET}")
    print(f"  - mIoU:           {seg_results['mIoU']:.4f}")
    print(f"  - Pixel Accuracy: {seg_results['pixel_accuracy']:.4f}")
    print("\n  Class-wise Intersection over Union (IoU):")
    for i, class_name in enumerate(CLASS_NAMES_SEG):
        iou = seg_results['iou_per_class'][i]
        print(f"    * {class_name:<25}: {iou:.4f}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADAS Multi-Task Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", type=str, default="checkpoints/best.pt", help="Path to model weights (.pt)")
    args = parser.parse_args()
    
    evaluate(args.config, args.weights)
