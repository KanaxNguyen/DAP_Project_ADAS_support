"""
src/tracker.py – Greedy IoU Tracker cho Biển Báo Giao Thông
=============================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    Theo dõi biển báo giao thông qua nhiều frame bằng thuật toán
    Greedy IoU Matching (ghim cặp tạm thời tốt nhất theo IoU).

Lưu ý quan trọng:
    Đây là triển khai TỰ VIẼT, không dùng thư viện ByteTrack hay SORT gốc.
    Các ý tưởng chính (track age, min_hits, IoU matching) được lấy cảm hứng từ:

Nguồn tham khảo:
    [1] Bewley et al., "Simple Online and Realtime Tracking" (SORT), ICIP 2016.
        track age, min_hits, IoU matching mechanism.
    [2] Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
        ECCV 2022. Ý tưởng phân tích biển báo theo điểm tin cậy và liên kết IoU.

Lưu ý: Code tự viết hoàn toàn (from scratch).
"""
import numpy as np

class Track:
    def __init__(self, track_id, bbox, confidence, class_id):
        self.id = track_id
        self.bbox = bbox          # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.age = 0              # frames since last match
        self.hits = 1             # total frames matched

    def update(self, bbox, confidence, class_id):
        # Weighted bounding box smoothing (e.g., 0.8 * old + 0.2 * new)
        self.bbox = 0.7 * np.array(self.bbox) + 0.3 * np.array(bbox)
        self.confidence = confidence
        self.class_id = class_id
        self.age = 0
        self.hits += 1

def calculate_iou(boxA, boxB):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

class SignTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, confidence, class_id]
        Returns:
            list of active and stable Track objects.
        """
        # Increment age for all existing tracks
        for track in self.tracks:
            track.age += 1

        # Match detections to existing tracks
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(detections) > 0:
            # Create IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    # We match only if they are the same class or both are traffic signs
                    # For signs, class can change slightly due to noise, but same class matching is safer.
                    # We allow matching between same classes or general proximity if needed, but same class is robust.
                    if int(track.class_id) == int(det[5]):
                        iou_matrix[t_idx, d_idx] = calculate_iou(track.bbox, det[:4])
                    else:
                        iou_matrix[t_idx, d_idx] = 0.0

            # Greedy matching
            while True:
                # Find max IoU value
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break

                # Get index of max value
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

                # Match found
                self.tracks[t_idx].update(detections[d_idx][:4], detections[d_idx][4], detections[d_idx][5])
                
                # Mark as matched
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)

                # Zero out corresponding row and column in iou_matrix
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(self.next_id, det[:4], det[4], det[5])
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks (age > max_age)
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]

        # Return only stable active tracks (hits >= min_hits and not stale)
        stable_tracks = [track for track in self.tracks if track.hits >= self.min_hits and track.age == 0]
        return stable_tracks
