"""
Kiểm thử bộ suy luận tốc độ theo Thông tư 38/2024/TT-BGTVT.
Chạy: python -m pytest tests/  (hoặc python tests/test_legal_speed_inference.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.legal_speed_inference import LegalSpeedInference  # noqa: E402

P127_80, R420_ENTER, R421_EXIT = 3, 5, 6


class FakeTrack:
    def __init__(self, class_id, bbox=(0, 0, 10, 10)):
        self.class_id = class_id
        self.bbox = bbox


DIVIDER = np.zeros((100, 100), dtype=int)
DIVIDER[:5, :5] = 2  # 0.25% pixel là dải phân cách, trên ngưỡng mặc định
NO_DIVIDER = np.zeros((100, 100), dtype=int)


def run(steps):
    model = LegalSpeedInference(smoothing_window=1)
    return [model.infer_speed_limit(tracks, mask)[0] for tracks, mask in steps]


def test_outside_residential_defaults():
    assert run([([], NO_DIVIDER)]) == [80]
    assert run([([], DIVIDER)]) == [90]


def test_inside_residential_defaults():
    assert run([([FakeTrack(R420_ENTER)], NO_DIVIDER), ([], NO_DIVIDER)]) == [50, 50]
    assert run([([FakeTrack(R420_ENTER)], DIVIDER), ([], DIVIDER)]) == [60, 60]


def test_leaving_residential_zone():
    assert run([([FakeTrack(R420_ENTER)], NO_DIVIDER), ([FakeTrack(R421_EXIT)], DIVIDER)]) == [50, 90]


def test_explicit_sign_overrides_and_persists():
    assert run([([FakeTrack(P127_80)], DIVIDER), ([], NO_DIVIDER)]) == [80, 80]


def test_majority_voting_delays_switch():
    model = LegalSpeedInference(smoothing_window=10)
    speeds = [model.infer_speed_limit([], DIVIDER)[0] for _ in range(6)]
    assert speeds == [80, 80, 80, 80, 80, 90]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All tests passed.")
