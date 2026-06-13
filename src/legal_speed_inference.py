"""
src/legal_speed_inference.py – Bộ Suy Luận Giới Hạn Tốc Độ (Luật Việt Nam)
==========================================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Suy luận giới hạn tốc độ hợp pháp dựa trên:
    - Biển báo P.127 (giới hạn tốc độ cụ thể)
    - Biển báo R.420 (vào khu dân cư) / R.421 (ra khỏi khu dân cư)
    - Phân vùng dải phân cách cứng (hard divider) từ segmentation mask
    Sử dụng bầu chọn đa số (majority voting) theo cửa sổ lịch sử để làm mượt kết quả.

Cơ sở pháp lý:
    - Luật Giao Thông Đường Bộ Việt Nam 2008 (sửa đổi 2018).
    - Thông tư 31/2019/TT-BGTVT về quy định tốc độ xe cơ giới.

Lưu ý: Lô-gic suy luận là hủ thống quy tắc (rule-based) do nhóm tự thiết kế.
"""
import torch
import numpy as np
from collections import Counter

class LegalSpeedInference:
    def __init__(self, smoothing_window=10, hard_divider_threshold=0.0005):
        """
        smoothing_window: Number of past frames to smooth the speed limit over.
        hard_divider_threshold: Ratio of hard_divider pixels to total pixels to count as present.
        """
        self.smoothing_window = smoothing_window
        self.hard_divider_threshold = hard_divider_threshold
        
        # Initial states
        self.current_speed = 60  # Default legal speed in Vietnam (non-residential, undivided)
        self.residential_state = False  # False: outside, True: inside
        self.last_explicit_speed = None  # Holds speed from P127 if active
        
        # History queue for temporal smoothing
        self.speed_history = [self.current_speed] * self.smoothing_window

        # Class maps
        self.p127_classes = {
            0: 40,   # P127_40
            1: 50,   # P127_50
            2: 60,   # P127_60
            3: 80,   # P127_80
            4: 90    # P127_90
        }
        self.R420_ENTER = 5
        self.R421_EXIT = 6

    def infer_speed_limit(self, stable_tracks, seg_mask):
        """
        stable_tracks: List of active Track objects from the tracker.
        seg_mask: Segmentation mask tensor [H, W] or numpy array.
        Returns:
            final_speed: Smoothed speed limit (int)
            reason: String description of how the speed limit was determined
        """
        # 1. Check for hard divider in segmentation mask
        if isinstance(seg_mask, torch.Tensor):
            divider_pixels = (seg_mask == 2).sum().item()
            total_pixels = seg_mask.numel()
        else:
            divider_pixels = np.sum(seg_mask == 2)
            total_pixels = seg_mask.size
            
        divider_ratio = divider_pixels / total_pixels
        has_hard_divider = divider_ratio > self.hard_divider_threshold
        
        # 2. Parse traffic signs in the current frame
        active_p127_speeds = []
        enter_residential_detected = False
        exit_residential_detected = False
        
        # We sort tracks by area (box width * height) to process the closest signs first
        # bbox is [x1, y1, x2, y2]
        sorted_tracks = sorted(
            stable_tracks, 
            key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]), 
            reverse=True
        )
        
        for track in sorted_tracks:
            class_id = int(track.class_id)
            if class_id in self.p127_classes:
                active_p127_speeds.append(self.p127_classes[class_id])
            elif class_id == self.R420_ENTER:
                enter_residential_detected = True
            elif class_id == self.R421_EXIT:
                exit_residential_detected = True

        reason = "Mặc định (Không có biển báo)"

        # 3. Determine Speed Target for Current Frame
        # Case A: Explicit P127 Speed sign detected
        if len(active_p127_speeds) > 0:
            # Take the closest (largest) speed sign
            self.last_explicit_speed = active_p127_speeds[0]
            speed_target = self.last_explicit_speed
            reason = f"Phát hiện biển giới hạn tốc độ P.127 ({speed_target} km/h)"
            
        # Case B: Residential Zone enters
        elif enter_residential_detected:
            self.residential_state = True
            self.last_explicit_speed = None  # Zone entry clears prior explicit speeds
            speed_target = 50
            reason = "Vào khu dân cư (R.420) -> Giới hạn 50 km/h"
            
        # Case C: Residential Zone exits
        elif exit_residential_detected:
            self.residential_state = False
            self.last_explicit_speed = None  # Zone exit clears prior explicit speeds
            if has_hard_divider:
                speed_target = 80
                reason = "Ra khỏi khu dân cư (R.421) + Có dải phân cách -> Giới hạn 80 km/h"
            else:
                speed_target = 60
                reason = "Ra khỏi khu dân cư (R.421) + Không có dải phân cách -> Giới hạn 60 km/h"
                
        # Case D: No signs detected (carry over states or infer from context)
        else:
            if self.last_explicit_speed is not None:
                # Keep showing the explicit speed limit sign
                speed_target = self.last_explicit_speed
                reason = f"Theo biển báo tốc độ trước đó P.127 ({speed_target} km/h)"
            else:
                # Infer from context
                if self.residential_state:
                    speed_target = 50
                    reason = "Đang trong khu dân cư (R.420) -> Giới hạn 50 km/h"
                else:
                    if has_hard_divider:
                        speed_target = 80
                        reason = "Ngoài khu dân cư + Có dải phân cách -> Giới hạn 80 km/h"
                    else:
                        speed_target = 60
                        reason = "Ngoài khu dân cư + Không có dải phân cách -> Giới hạn 60 km/h"

        # 4. Temporal Smoothing / Bayesian-like Voting
        self.speed_history.append(speed_target)
        if len(self.speed_history) > self.smoothing_window:
            self.speed_history.pop(0)
            
        # Get the mode (most common speed limit in our window)
        counter = Counter(self.speed_history)
        final_speed = counter.most_common(1)[0][0]
        
        # If the smoothed speed limit matches self.current_speed, we keep the reason.
        # Otherwise, update reason to reflect the transition.
        if final_speed != speed_target:
            reason = f"Đang cập nhật... (Trạng thái tạm thời: {speed_target} km/h)"

        self.current_speed = final_speed
        return final_speed, reason
