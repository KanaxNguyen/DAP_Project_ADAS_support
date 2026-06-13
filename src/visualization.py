"""
src/visualization.py – HUD Overlay & Hiển Thị Kết Quả
=====================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Vẽ overlay HUD trên frame video:
    - Bounding box của biển báo (màu phân biệt theo class)
    - Segmentation mask (bề mặt trong suốt màu)
    - Hình HUD hiển thị tốc độ pháp lý hiện tại

Lưu ý: Đây là code tự viết bằng OpenCV, không dùng thư viện nào khác.
"""
import cv2
import numpy as np

# Color definitions (BGR)
COLORS_SEG = {
    1: (0, 220, 220),   # Lane marking: Light Yellow
    2: (0, 0, 255),     # Hard divider: Red
    3: (0, 255, 100)    # Drivable area: Semi-neon Green
}

# Sign color definitions
COLOR_SPEED_SIGN = (0, 0, 255)       # Red for P127
COLOR_ZONE_SIGN = (255, 100, 0)      # Blue/Cyan for R420/R421

CLASS_NAMES_DET = {
    0: "P.127 - 40km/h",
    1: "P.127 - 50km/h",
    2: "P.127 - 60km/h",
    3: "P.127 - 80km/h",
    4: "P.127 - 90km/h",
    5: "R.420 - Vao KDC",
    6: "R.421 - Ra KDC"
}

def draw_hud(frame, speed_limit, speed_reason):
    """
    Renders a premium semi-transparent dashboard HUD on the top of the frame.
    """
    h, w = frame.shape[:2]
    
    # 1. Draw top panel background (semi-transparent black)
    hud_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 15, 15), -1)
    # Blend overlay for transparent effect
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw a divider line under HUD
    cv2.line(frame, (0, hud_h), (w, hud_h), (80, 80, 80), 2)
    
    # 2. Draw Speed Limit Sign (Standard Circular Red Boundary)
    cx, cy = 60, hud_h // 2
    r_outer = 38
    # Red circle border
    cv2.circle(frame, (cx, cy), r_outer, (0, 0, 255), 6, cv2.LINE_AA)
    # White filling
    cv2.circle(frame, (cx, cy), r_outer - 3, (255, 255, 255), -1, cv2.LINE_AA)
    
    # Draw limit text
    font = cv2.FONT_HERSHEY_DUPLEX
    text = str(speed_limit)
    # Calculate text size to center it
    text_size = cv2.getTextSize(text, font, 1.1, 3)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(frame, text, (tx, ty), font, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    
    # 3. Draw Information Labels
    # Title
    cv2.putText(frame, "ADAS CONTEXT-AWARE SYSTEM (VIETNAM)", (130, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Info
    cv2.putText(frame, f"TOC DO LEGAL: {speed_limit} km/h", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TRANG THAI: {speed_reason}", (130, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    
    # System Status Indicator
    status_text = "ACTIVE"
    s_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (w - 110, 30), (w - 20, 60), (0, 150, 0), -1)
    cv2.putText(frame, status_text, (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def visualize_frame(frame, detections, seg_mask, speed_limit, speed_reason):
    """
    Overlays segmentation masks and bounding boxes on the frame.
    frame: np.array of shape [H, W, 3] (BGR)
    detections: list of tracks/detections: [x1, y1, x2, y2, score, class_id, (optional) track_id]
                Or Track objects.
    seg_mask: np.array of shape [H, W] (class IDs: 0-3)
    speed_limit: current legal speed (int)
    speed_reason: string description
    """
    h, w = frame.shape[:2]
    
    # 1. Overlay Segmentation Masks
    seg_overlay = np.zeros_like(frame)
    for class_id, color in COLORS_SEG.items():
        class_mask = (seg_mask == class_id)
        seg_overlay[class_mask] = color
        
    # Blend with original frame (alpha=0.65 for frame, beta=0.35 for mask overlay)
    mask = (seg_mask > 0)
    frame[mask] = cv2.addWeighted(frame, 0.65, seg_overlay, 0.35, 0)[mask]
    
    # 2. Draw Bounding Boxes of Signs
    for det in detections:
        # Can accept both lists/numpy arrays or Track objects
        if hasattr(det, 'bbox'):  # It's a Track object
            x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
            score = det.confidence
            class_id = int(det.class_id)
            track_id = det.id
        else:  # It's a list/numpy array
            x1, y1, x2, y2 = [int(coord) for coord in det[:4]]
            score = det[4]
            class_id = int(det[5])
            track_id = int(det[6]) if len(det) > 6 else None

        # Choose box color based on sign class
        color = COLOR_SPEED_SIGN if class_id <= 4 else COLOR_ZONE_SIGN
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label string
        cls_name = CLASS_NAMES_DET.get(class_id, "Sign")
        label = f"{cls_name} ({score:.2f})"
        if track_id is not None:
            label = f"ID:{track_id} {label}"
            
        # Draw background label rect
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Put text above box if space is available, else inside
        text_y = max(y1 - 10, 15)
        cv2.rectangle(frame, (x1, text_y - t_h - 4), (x1 + t_w, text_y + 4), color, -1)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 3. Draw HUD
    frame = draw_hud(frame, speed_limit, speed_reason)
    
    return frame
