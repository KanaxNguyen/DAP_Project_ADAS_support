"""
infer.py – Script Demo Inference trên Video / Webcam
=====================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Chạy vídu:
    python infer.py --source video.mp4 --weights checkpoints/best.pt --config configs/config.yaml
    python infer.py --source 0          # webcam
    python infer.py --source video.mp4 --sahi  # SAHI sliced inference
Điều phối:
    - Chạy mô hình ADAS trên video hoặc webcam.
    - Hiển thị kết quả dưới dạng HUD overlay (biển báo, mask, tốc độ).
    - Lưu video kết quả vào runs/output.mp4.
"""
import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from src.model import ADASMultiTaskNetwork
from src.tracker import SignTracker
from src.legal_speed_inference import LegalSpeedInference
from src.sahi_inference import sahi_sliced_inference
from src.utils import non_max_suppression

def create_synthetic_video(file_path, num_frames=90):
    """
    Creates a synthetic road video with lane markings, a hard divider,
    and approaching traffic signs to demonstrate the ADAS logic.
    """
    print(f"Creating synthetic video '{file_path}' (90 frames)...")
    width, height = 1280, 720
    # Use MP4V or MJPG codec for compatibility across systems
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))
    
    for frame_idx in range(num_frames):
        # 1. Base image (sky and grassy background)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky (blue)
        frame[0:350, :] = [230, 200, 150]
        # Ground (green)
        frame[350:, :] = [70, 130, 50]
        
        # 2. Road (gray polygon)
        road_pts = np.array([[200, height], [1080, height], [640, 350]], np.int32)
        cv2.fillPoly(frame, [road_pts], (80, 80, 80))
        
        # 3. Hard divider (left barrier, red/white blocks)
        # Draws a dividing strip on the left side of the road
        divider_pts = np.array([[500, height], [530, height], [630, 350], [625, 350]], np.int32)
        cv2.fillPoly(frame, [divider_pts], (220, 220, 220))
        
        # 4. Moving lane markings (white dashed lines)
        # Shift lines down based on frame index to simulate movement
        offset = (frame_idx * 15) % 120
        for y_dash in range(350 + offset, height, 120):
            # Compute perspective interpolation for width and position
            t = (y_dash - 350) / (height - 350)
            y_start = y_dash
            y_end = min(y_start + 40, height)
            
            x_center = int(640 + t * 200) # center of right lane
            w_dash = int(4 + t * 12)
            
            cv2.line(frame, (x_center, y_start), (x_center, y_end), (255, 255, 255), w_dash)

        # 5. Approaching Traffic Signs
        # We simulate:
        # Frames 0-40: Entering residential (R420) sign approaching on the right side
        # Frames 45-90: Speed limit 50 (P127_50) sign approaching on the right side
        if frame_idx < 40:
            # R420 sign approaching
            t = frame_idx / 40.0
            size = int(20 + t * 80)
            sx = int(850 + t * 250)
            sy = int(320 - t * 50)
            
            # Draw blue sign post
            cv2.line(frame, (sx + size//2, sy + size), (sx + size//2, sy + size + 80), (100, 100, 100), int(2 + t*6))
            # Draw blue rect (R420 enter residential)
            cv2.rectangle(frame, (sx, sy), (sx + size, sy + size), (255, 120, 0), -1) # Blue in BGR
            cv2.rectangle(frame, (sx, sy), (sx + size, sy + size), (255, 255, 255), int(1 + t*3))
            # Inner white graphic representation
            cv2.circle(frame, (sx + size//2, sy + size//3), size//5, (255, 255, 255), -1)
            
        elif frame_idx >= 45:
            # P127_50 speed sign approaching
            t = (frame_idx - 45) / 45.0
            size = int(20 + t * 80)
            sx = int(850 + t * 250)
            sy = int(320 - t * 50)
            
            # Draw sign post
            cv2.line(frame, (sx + size//2, sy + size), (sx + size//2, sy + size + 80), (100, 100, 100), int(2 + t*6))
            # Outer red circle
            cv2.circle(frame, (sx + size//2, sy + size//2), size//2, (0, 0, 255), -1)
            # Inner white circle
            cv2.circle(frame, (sx + size//2, sy + size//2), int(size//2 * 0.8), (255, 255, 255), -1)
            # Text '50'
            cv2.putText(frame, "50", (sx + int(size * 0.25), sy + int(size * 0.65)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2 + t * 0.9, (0, 0, 0), int(1 + t * 2), cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()
    print("Synthetic video created.")

def run_inference(config_path, weights_path, source, sahi_override=None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config['model']
    tracker_cfg = config['tracker']
    speed_cfg = config['speed_inference']
    infer_cfg = config['inference']
    
    use_sahi = sahi_override if sahi_override is not None else infer_cfg['sahi']['enabled']
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    model = ADASMultiTaskNetwork(
        backbone_name=model_cfg['backbone'],
        grid_size=model_cfg['grid_size'],
        num_classes_det=model_cfg['num_classes_det'],
        num_classes_seg=model_cfg['num_classes_seg'],
        pretrained=False
    ).to(device)
    
    # Load weights
    if not os.path.exists(weights_path):
        print(f"Error: Checkpoint {weights_path} not found. Running with randomly initialized weights (ONLY FOR TEST).")
    else:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights successfully from {weights_path}")
        
    model.eval()
    
    # Instantiate Tracker & Speed limits logic
    tracker = SignTracker(
        max_age=tracker_cfg['max_age'],
        min_hits=tracker_cfg['min_hits'],
        iou_threshold=tracker_cfg['iou_threshold']
    )
    
    speed_engine = LegalSpeedInference(
        smoothing_window=speed_cfg['smoothing_window'],
        hard_divider_threshold=speed_cfg['hard_divider_threshold']
    )
    
    # Handle synthetic video fallback
    if source == "video.mp4" and not os.path.exists(source):
        create_synthetic_video("video.mp4")
        
    # Open Video Source
    # Check if webcam '0' or path
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_video = True
    else:
        cap = cv2.VideoCapture(source)
        is_video = cap.isOpened()
        
    # Create directories for outputs
    os.makedirs("runs", exist_ok=True)
    os.makedirs("runs/images", exist_ok=True)
    
    video_writer = None
    output_video_path = "runs/output.mp4"
    
    # Importing viz components
    from src.visualization import visualize_frame
    
    frame_idx = 0
    print(f"\nProcessing source: {source}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h_orig, w_orig = frame.shape[:2]
        
        # 1. MODEL INFERENCE
        if use_sahi:
            # Run SAHI sliced inference on the high-res frame
            detections, seg_mask = sahi_sliced_inference(
                model=model,
                image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                slice_size=infer_cfg['sahi']['slice_size'],
                overlap_ratio=infer_cfg['sahi']['overlap_ratio'],
                conf_threshold=infer_cfg['conf_threshold'],
                iou_threshold=infer_cfg['iou_threshold'],
                device=device
            )
        else:
            # Regular single-scale inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (infer_cfg['sahi']['slice_size'], infer_cfg['sahi']['slice_size']))
            
            # Prepare tensor
            img_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                det_out, seg_out = model(img_tensor)
                
                # Bbox NMS
                detections_list = non_max_suppression(
                    det_out, conf_thres=infer_cfg['conf_threshold'], iou_thres=infer_cfg['iou_threshold']
                )
                raw_dets = detections_list[0].cpu().numpy()
                
                # Scale boxes back to original size
                detections = []
                for det in raw_dets:
                    x1 = det[0] * w_orig
                    y1 = det[1] * h_orig
                    x2 = det[2] * w_orig
                    y2 = det[3] * h_orig
                    detections.append([x1, y1, x2, y2, det[4], det[5]])
                detections = np.array(detections, dtype=np.float32) if len(detections) > 0 else np.zeros((0, 6))
                
                # Segmentation argmax and resize back to original size
                seg_probs = torch.softmax(seg_out, dim=1)
                seg_mask = torch.argmax(seg_probs[0], dim=0).cpu().numpy().astype(np.uint8)
                seg_mask = cv2.resize(seg_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                
        # 2. TRACKING
        stable_tracks = tracker.update(detections)
        
        # 3. SPEED LIMIT INFERENCE
        speed_limit, reason = speed_engine.infer_speed_limit(stable_tracks, seg_mask)
        
        # 4. VISUALIZATION
        annotated_frame = visualize_frame(
            frame=frame,
            detections=stable_tracks,
            seg_mask=seg_mask,
            speed_limit=speed_limit,
            speed_reason=reason
        )
        
        # 5. WRITE/SAVE OUTPUT
        if is_video:
            if video_writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w_orig, h_orig))
                print(f"Saving output video to: {output_video_path}")
            video_writer.write(annotated_frame)
        
        # Save every 15th frame as a demo image
        if frame_idx % 15 == 0:
            img_out_path = f"runs/images/frame_{frame_idx:04d}.jpg"
            cv2.imwrite(img_out_path, annotated_frame)
            
        frame_idx += 1
        
        # Display window (only works in interactive desktop GUI environments)
        try:
            cv2.imshow("ADAS Perception HUD", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            # Fallback for environments with no display head
            pass
            
    cap.release()
    if video_writer is not None:
        video_writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
        
    print(f"\nInference completed successfully. Processed {frame_idx} frames.")
    print(f"Output video saved to: {output_video_path}")
    print("Demo frames saved in: runs/images/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Multi-Task Inference script")
    parser.add_argument("--source", type=str, default="video.mp4", help="Video file path, image folder path, or '0' for webcam")
    parser.add_argument("--weights", type=str, default="checkpoints/best.pt", help="Path to checkpoint weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--sahi", action="store_true", help="Force SAHI slicing inference")
    args = parser.parse_args()
    
    run_inference(args.config, args.weights, args.source, sahi_override=args.sahi if args.sahi else None)
