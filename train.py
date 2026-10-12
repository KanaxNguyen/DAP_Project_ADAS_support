"""
train.py – Script Huấn Luyện Mô Hình ADAS Đa Nhiệm
=====================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Chạy vídu: python train.py --config configs/config.yaml
Điều phối:
    - Tự động tạo dữ liệu tổng hợp nếu không có data thực.
    - Huấn luyện ADASMultiTaskNetwork với AdamW + CosineAnnealingLR.
    - Lưu checkpoint tốt nhất (best.pt) và cuối cùng (last.pt).
"""
import os
import argparse
import yaml
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset import get_dataloader
from src.model import ADASMultiTaskNetwork
from src.losses import ADASMultiTaskLoss
from src.utils import non_max_suppression
from src.metrics import evaluate_detection, SegmentationMetricTracker

def setup_synthetic_data():
    """
    Creates a synthetic toy dataset if no training data is found.
    Ensures the code runs immediately out-of-the-box.
    """
    print("Checking dataset directories...")
    paths = [
        "data/images/train", "data/labels/train", "data/masks/train",
        "data/images/val", "data/labels/val", "data/masks/val"
    ]
    
    # Check if any path is missing or empty
    data_exists = True
    for p in paths:
        if not os.path.exists(p) or len(os.listdir(p)) == 0:
            data_exists = False
            break
            
    if data_exists:
        print("Dataset found. Ready to train.")
        return

    print("No dataset detected. Creating synthetic toy dataset for validation and execution testing...")
    import cv2
    for p in paths:
        os.makedirs(p, exist_ok=True)
        
    # Generate 8 training images and 4 validation images
    for mode, count in [("train", 8), ("val", 4)]:
        for i in range(count):
            img_name = f"toy_{mode}_{i}"
            
            # 1. Generate Image (640x640x3)
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            # Add road background (dark gray triangle)
            pts = np.array([[100, 640], [540, 640], [320, 300]], np.int32)
            cv2.fillPoly(img, [pts], (40, 40, 40))
            # Draw a simulated speed sign (red circle with white center)
            cv2.circle(img, (200, 200), 20, (0, 0, 255), -1)
            cv2.circle(img, (200, 200), 16, (255, 255, 255), -1)
            cv2.putText(img, "50", (190, 207), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            
            # Save image
            cv2.imwrite(f"data/images/{mode}/{img_name}.jpg", img)
            
            # 2. Generate YOLO Label (Class 1: P127_50 at center (200, 200) relative coordinates)
            # x_center = 200/640 = 0.3125, y_center = 200/640 = 0.3125
            # width = 40/640 = 0.0625, height = 40/640 = 0.0625
            with open(f"data/labels/{mode}/{img_name}.txt", "w") as f:
                f.write("1 0.3125 0.3125 0.0625 0.0625\n")
                
            # 3. Generate Mask (0: background, 1: lane_marking, 2: hard_divider, 3: drivable_area)
            mask = np.zeros((640, 640), dtype=np.uint8)
            # Drivable area inside the triangle
            cv2.fillPoly(mask, [pts], 3)
            # Lane markings (left and right lines)
            cv2.line(mask, (320, 300), (100, 640), 1, 8)
            cv2.line(mask, (320, 300), (540, 640), 1, 8)
            # Hard divider in center (thin vertical line on horizon)
            cv2.line(mask, (320, 300), (320, 450), 2, 4)
            
            # Save mask
            cv2.imwrite(f"data/masks/{mode}/{img_name}.png", mask)
            
    print(f"Synthetic dataset created successfully: {count} images in data/.")

def train(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    setup_synthetic_data()
    
    # Paths & Hyperparams
    train_cfg = config['train']
    model_cfg = config['model']
    data_cfg = config['data']
    
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() and train_cfg['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataloaders
    train_loader = get_dataloader(
        img_dir=data_cfg['train_images'],
        label_dir=data_cfg['train_labels'],
        mask_dir=data_cfg['train_masks'],
        img_size=data_cfg['image_size'],
        grid_size=model_cfg['grid_size'],
        num_classes_det=model_cfg['num_classes_det'],
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
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
        pretrained=True
    ).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = ADASMultiTaskLoss(
        grid_size=model_cfg['grid_size'],
        num_classes_det=model_cfg['num_classes_det'],
        num_classes_seg=model_cfg['num_classes_seg']
    )
    
    optimizer = AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])
    
    # Save Directories
    save_dir = train_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print("\nStarting Training Loop...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_loss_details = {}
        
        for batch_idx, (images, masks, encoded_dets, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            encoded_dets = encoded_dets.to(device)
            
            optimizer.zero_grad()
            det_out, seg_out = model(images)
            
            loss, loss_details = criterion(det_out, seg_out, encoded_dets, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for k, v in loss_details.items():
                train_loss_details[k] = train_loss_details.get(k, 0.0) + v
                
        # Average loss
        num_batches = len(train_loader)
        train_loss /= num_batches
        for k in train_loss_details:
            train_loss_details[k] /= num_batches
            
        scheduler.step()
        
        # --- VALIDATION PHASE ---
        if epoch % train_cfg['val_interval'] == 0 or epoch == train_cfg['epochs']:
            model.eval()
            val_loss = 0.0
            val_loss_details = {}
            
            # Metrics Trackers
            seg_tracker = SegmentationMetricTracker(num_classes=model_cfg['num_classes_seg'])
            pred_boxes_all = []
            target_boxes_all = []
            
            with torch.no_grad():
                for images, masks, encoded_dets, raw_boxes in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    encoded_dets = encoded_dets.to(device)
                    
                    det_out, seg_out = model(images)
                    loss, loss_details = criterion(det_out, seg_out, encoded_dets, masks)
                    
                    val_loss += loss.item()
                    for k, v in loss_details.items():
                        val_loss_details[k] = val_loss_details.get(k, 0.0) + v
                        
                    # Segmentation metric updates
                    seg_probs = torch.softmax(seg_out, dim=1)
                    pred_seg_masks = torch.argmax(seg_probs, dim=1).cpu().numpy()
                    target_seg_masks = masks.cpu().numpy()
                    seg_tracker.update(pred_seg_masks, target_seg_masks)
                    
                    # Detection metric updates
                    # Decode prediction outputs using NMS
                    detections = non_max_suppression(det_out, conf_thres=0.1, iou_thres=0.45)
                    for det, raw_box in zip(detections, raw_boxes):
                        pred_boxes_all.append(det.cpu().numpy())
                        target_boxes_all.append(raw_box.cpu().numpy())
                        
            val_loss /= len(val_loader)
            for k in val_loss_details:
                val_loss_details[k] /= len(val_loader)
                
            # Compute final metrics
            seg_metrics = seg_tracker.get_results()
            det_metrics = evaluate_detection(
                pred_boxes_all, target_boxes_all, num_classes=model_cfg['num_classes_det']
            )
            
            # Print epoch summary
            print(f"Epoch [{epoch}/{train_cfg['epochs']}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"mIoU: {seg_metrics['mIoU']:.4f} | mAP@0.5: {det_metrics['mAP']:.4f}")
            
            # Save Checkpoints
            # Save best according to validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Ép về float Python: số kiểu numpy làm torch.load(weights_only=True) báo lỗi từ PyTorch 2.6
                    'val_loss': float(val_loss),
                    'mIoU': float(seg_metrics['mIoU']),
                    'mAP': float(det_metrics['mAP'])
                }, os.path.join(save_dir, "best.pt"))
                print(f"--> Saved best model checkpoint to {os.path.join(save_dir, 'best.pt')} (Val Loss improved to {val_loss:.4f})")
                
            # Save last checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_loss)
            }, os.path.join(save_dir, "last.pt"))
            
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Multi-Task Training script")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    train(args.config)
