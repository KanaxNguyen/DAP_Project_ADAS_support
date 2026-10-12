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
    - Thông tư 38/2024/TT-BGTVT (hiệu lực 01/01/2025, thay Thông tư 31/2019/TT-BGTVT),
      Điều 6 (trong khu vực đông dân cư) và Điều 7 (ngoài khu vực đông dân cư).

Giả định:
    - Xe là ô tô chở người đến 30 chỗ (trừ xe buýt) hoặc ô tô tải đến 3,5 tấn.
    - Thấy dải phân cách cứng thì coi là "đường đôi". Đường một chiều từ hai làn trở lên
      cũng được mức của đường đôi, nhưng mô hình chưa nhận biết được nên bị tính như đường hai chiều.
    - Không áp dụng cho đường cao tốc (Điều 9).

Lưu ý: Lô-gic suy luận là hệ thống quy tắc (rule-based) do nhóm tự thiết kế.
"""
import torch
import numpy as np
from collections import Counter

# Tốc độ tối đa (km/h) theo Thông tư 38/2024/TT-BGTVT, khóa: (trong khu dân cư, có dải phân cách)
DEFAULT_SPEED_LIMITS = {
    (True, True): 60,    # Điều 6: đường đôi
    (True, False): 50,   # Điều 6: đường hai chiều
    (False, True): 90,   # Điều 7: đường đôi
    (False, False): 80,  # Điều 7: đường hai chiều
}

class LegalSpeedInference:
    def __init__(self, smoothing_window=10, hard_divider_threshold=0.0005):
        """
        smoothing_window: Number of past frames to smooth the speed limit over.
        hard_divider_threshold: Ratio of hard_divider pixels to total pixels to count as present.
        """
        self.smoothing_window = smoothing_window
        self.hard_divider_threshold = hard_divider_threshold

        # Initial states
        self.current_speed = DEFAULT_SPEED_LIMITS[(False, False)]  # Ngoài khu dân cư, đường hai chiều
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
        divider_text = "Có dải phân cách" if has_hard_divider else "Không có dải phân cách"

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
            speed_target = DEFAULT_SPEED_LIMITS[(True, has_hard_divider)]
            reason = f"Vào khu dân cư (R.420) + {divider_text} -> Giới hạn {speed_target} km/h"

        # Case C: Residential Zone exits
        elif exit_residential_detected:
            self.residential_state = False
            self.last_explicit_speed = None  # Zone exit clears prior explicit speeds
            speed_target = DEFAULT_SPEED_LIMITS[(False, has_hard_divider)]
            reason = f"Ra khỏi khu dân cư (R.421) + {divider_text} -> Giới hạn {speed_target} km/h"

        # Case D: No signs detected (carry over states or infer from context)
        else:
            if self.last_explicit_speed is not None:
                # Keep showing the explicit speed limit sign
                speed_target = self.last_explicit_speed
                reason = f"Theo biển báo tốc độ trước đó P.127 ({speed_target} km/h)"
            else:
                # Infer from context
                speed_target = DEFAULT_SPEED_LIMITS[(self.residential_state, has_hard_divider)]
                zone_text = "Trong khu dân cư" if self.residential_state else "Ngoài khu dân cư"
                reason = f"{zone_text} + {divider_text} -> Giới hạn {speed_target} km/h"

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
