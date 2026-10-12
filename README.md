# Vietnam ADAS – Đồ Án Phân Tích Dữ Liệu (Nhóm 6)

> **Tên đề tài:** Nhận Diện Biển Báo Tốc Độ & Phân Vùng Làn Đường Tích Hợp cho Hệ Thống ADAS Theo Bối Cảnh Giao Thông Việt Nam  
> **Bản chất:** Đây là **đồ án môn học** (Phân Tích Dữ Liệu – Kỳ 4) của **Nhóm 6**, không phải sản phẩm thương mại hay công bố khoa học chính thức.

Dự án xây dựng một **nguyên mẫu học thuật (prototype)** hệ thống hỗ trợ lái xe nâng cao (ADAS) đa nhiệm, tích hợp:
- Nhận diện biển báo giao thông (biển tốc độ P.127, biển khu dân cư R.420/R.421)
- Phân vùng đường (vạch kẻ đường, dải phân cách cứng, vùng lái xe)
- Suy luận tốc độ hợp pháp theo quy định giao thông đường bộ Việt Nam

---

## ⚠️ Tuyên Bố Về Nguồn (Attribution & References)

Dự án này là **nguyên mẫu học thuật tự xây dựng từ đầu** (from scratch), lấy cảm hứng và triển khai lại các kỹ thuật từ những công trình nghiên cứu đã công bố. **Không có code nào được sao chép trực tiếp** từ các repository gốc.

| Kỹ thuật trong dự án | Công trình tham khảo |
|----------------------|----------------------|
| **ResNet Backbone** (pretrained weights) | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. Trọng số pretrained tải qua `torchvision`. |
| **Feature Pyramid Network (FPN) Neck** | Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017. |
| **YOLO-style Grid Detection Head** | Redmon et al., *You Only Look Once* series (YOLOv1–v5); kiến trúc đầu phát hiện theo lưới ô vuông. |
| **Focal Loss** (objectness loss) | Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet), ICCV 2017. |
| **CIoU Loss** (bounding box regression) | Zheng et al., *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression*, AAAI 2020. |
| **Dice Loss** (segmentation) | Milletari et al., *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*, 3DV 2016. |
| **SAHI-style Sliced Inference** | Akyon et al., *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*, ICIP 2022. |
| **Greedy IoU Tracker** | Lấy cảm hứng từ SORT (Bewley et al., ICIP 2016) và ByteTrack (Zhang et al., ECCV 2022). Triển khai là Greedy IoU Matching **tự viết**, không dùng thư viện ByteTrack gốc. |

---

## 🌟 Tính Năng Của Nguyên Mẫu

1. **Multi-Task Neural Network**: Chia sẻ Encoder (ResNet) + FPN Neck cho hai đầu đầu ra:
   - **Detection Head**: Nhận diện 7 loại biển báo (P.127 các tốc độ, R.420, R.421) theo phong cách YOLO.
   - **Segmentation Head**: Phân vùng 4 lớp (nền, vạch kẻ đường, dải phân cách cứng, vùng lái xe).
2. **Greedy IoU Tracker**: Theo dõi biển báo qua nhiều frame giúp lọc nhiễu phát hiện nhất thời.
3. **Suy Luận Tốc Độ Hợp Pháp (Luật Việt Nam)**: Tự động áp dụng quy định tốc độ dựa trên biển báo và loại đường.
4. **Temporal Smoothing (Majority Voting)**: Bộ lọc bình chọn đa số qua cửa sổ lịch sử $N$ frame, tránh nhấp nháy tốc độ.
5. **SAHI-style Inference**: Hỗ trợ chia nhỏ frame để tăng độ chính xác với biển báo nhỏ ở xa.
6. **Dữ Liệu Tổng Hợp Tự Động**: Tự sinh dataset và video giả lập nếu chưa có dữ liệu thực.

---

## 📐 Kiến Trúc Mô Hình (Tóm Tắt)

```
Input Image [B, 3, 640, 640]
       │
  ResNet18/34 Backbone (pretrained ImageNet)
       │
  ┌────┴────────────────┐
  c2 (stride 8)  c3 (stride 16)  c4 (stride 32)
       └────────────────┴────────────────────────┐
                    FPN Neck                      │
          (top-down lateral connections)          │
       ┌────────────────────────────────────────┘
  p2 (stride 8)  p3 (stride 16)  p4 (stride 32)
       │                   │
  Seg Head             Det Head
  (4 classes,         (YOLO-style,
  CE+Dice Loss)        Focal+CIoU Loss)
       │                   │
  Seg Mask [H,W]    Det Grid [S,S, 5+C]
```

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
DAP_Project_ADAS_support/
│
├── configs/
│   └── config.yaml                 # File cấu hình siêu tham số, đường dẫn dữ liệu
│
├── data/                           # Thư mục chứa tập dữ liệu
│   ├── images/                     # Ảnh đầu vào (.jpg, .png)
│   ├── labels/                     # Nhãn bboxes dạng YOLO (.txt)
│   └── masks/                      # Ảnh mặt nạ segmentation dạng PNG (.png)
│
├── src/
│   ├── dataset.py                  # Dataloader tải ảnh, nhãn bboxes và masks
│   ├── model.py                    # Kiến trúc MTL: ResNet Backbone + FPN + Dual Heads
│   ├── losses.py                   # Tổ hợp loss: Focal, CIoU, Cross-Entropy, Dice Loss
│   ├── metrics.py                  # Cách tính mAP@0.5 (Detection) và mIoU (Segmentation)
│   ├── sahi_inference.py           # Slicing inference hỗ trợ phát hiện vật thể nhỏ
│   ├── tracker.py                  # Greedy IoU Tracker theo dõi biển báo qua các frame
│   ├── legal_speed_inference.py    # Bộ suy luận luật tốc độ Việt Nam & làm mượt kết quả
│   ├── visualization.py            # HUD overlay vẽ bounding box, mask và biển tốc độ
│   └── utils.py                    # Hàm tiện ích (NMS, đổi hệ tọa độ, tính IoU)
│
├── tests/
│   └── test_legal_speed_inference.py  # Kiểm thử bộ luật tốc độ
│
├── train.py                        # Script huấn luyện hệ thống
├── evaluate.py                     # Script đánh giá hiệu năng trên tập validation
├── infer.py                        # Script chạy demo trên video/webcam
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── LICENSE                         # Giấy phép MIT
└── README.md                       # Tài liệu hướng dẫn sử dụng (file này)
```

---

## 🚀 Hướng Dẫn Cài Đặt Môi Trường

Dự án yêu cầu **Python 3.10+** và **PyTorch**.

```bash
# 1. Di chuyển vào thư mục repo
cd DAP_Project_ADAS_support

# 2. Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

---

## 📊 Chuẩn Bị Dataset Format

Nếu bạn muốn huấn luyện trên dữ liệu tự chuẩn bị, hãy sắp xếp cấu trúc thư mục như sau:

* **Images (`data/images/`)**: Đặt ảnh huấn luyện trong `train/` và ảnh đánh giá trong `val/`.
* **Labels (`data/labels/`)**: Chứa nhãn bounding boxes dạng YOLO `.txt` có tên trùng khớp với ảnh:
  `class_id x_center y_center width height` (tọa độ chuẩn hóa trong khoảng `[0, 1]`).
  
  **Danh sách các class biển báo (class_id từ 0 -> 6):**
  - `0`: P127_40 (Giới hạn 40 km/h)
  - `1`: P127_50 (Giới hạn 50 km/h)
  - `2`: P127_60 (Giới hạn 60 km/h)
  - `3`: P127_80 (Giới hạn 80 km/h)
  - `4`: P127_90 (Giới hạn 90 km/h)
  - `5`: R420_enter_residential (Bắt đầu khu đông dân cư)
  - `6`: R421_exit_residential (Hết khu đông dân cư)

* **Masks (`data/masks/`)**: Chứa ảnh mặt nạ phân vùng dạng `.png` đơn kênh (grayscale) trùng tên với ảnh. Giá trị của từng pixel tương ứng với class_id:
  - `0`: Nền (background)
  - `1`: Vạch kẻ đường (lane_marking)
  - `2`: Dải phân cách cứng (hard_divider)
  - `3`: Vùng xe di chuyển được (drivable_area)

> **💡 Lưu ý:** Nếu thư mục `data/` trống, hệ thống sẽ tự động tạo dữ liệu đồ họa mô phỏng (Toy Dataset) để bạn có thể chạy thử mã nguồn ngay lập tức. Dữ liệu này chỉ để kiểm tra pipeline chạy được, không dùng để đánh giá mô hình.

**Dữ liệu của nhóm:** ngoài dữ liệu công khai, nhóm tự thu thập và gán nhãn **300 ảnh đường Việt Nam** (vạch kẻ đường và khu dân cư) để bù các trường hợp dữ liệu công khai thiếu. Ảnh không kèm trong repo; `data/` được loại khỏi Git.

---

## 🛠️ Hướng Dẫn Sử Dụng

### 1. Huấn luyện (Train)
Chạy script `train.py` để thực hiện huấn luyện mô hình MTL. Nếu chưa có dữ liệu thực, mô hình sẽ tự tạo dữ liệu giả lập để chạy thử:
```bash
python train.py --config configs/config.yaml
```
Mô hình tốt nhất (dựa trên Validation Loss) và mô hình cuối cùng sẽ được lưu tại thư mục `checkpoints/best.pt` và `checkpoints/last.pt`.

### 2. Đánh giá (Evaluate)
Đánh giá chất lượng của mô hình trên tập validation (tính toán mAP@0.5 cho Detection và mIoU cho Segmentation):
```bash
python evaluate.py --config configs/config.yaml --weights checkpoints/best.pt
```

### 3. Suy luận Demo trên Video (Inference)
Chạy suy luận trên video đầu vào. Nếu bạn chưa có file `video.mp4` trong máy, script sẽ tự động tạo một video giả lập đường phố 3D dạng đơn giản (`video.mp4`) để chạy thử:
```bash
python infer.py --source video.mp4 --weights checkpoints/best.pt --config configs/config.yaml
```
* **Sử dụng SAHI (Slicing)** để phát hiện biển báo nhỏ ở xa:
  ```bash
  python infer.py --source video.mp4 --weights checkpoints/best.pt --config configs/config.yaml --sahi
  ```
* **Sử dụng Webcam làm nguồn:**
  ```bash
  python infer.py --source 0 --weights checkpoints/best.pt --config configs/config.yaml
  ```
* **Kết quả:**
  - Video HUD hoàn chỉnh được lưu tại: `runs/output.mp4`.
  - Một số frame mẫu trực quan được lưu tại thư mục: `runs/images/`.

---

## ⚙️ Logic Đề Xuất Tốc Độ Tối Đa (Luật Việt Nam)

Mô-đun `legal_speed_inference.py` áp dụng **Thông tư 38/2024/TT-BGTVT** (hiệu lực 01/01/2025, thay Thông tư 31/2019), Điều 6 và Điều 7:

| Tình huống | Tốc độ áp dụng |
|-----------|----------------|
| Phát hiện biển P.127 (bất kỳ) | Tốc độ trên biển (ưu tiên cao nhất) |
| Trong khu dân cư (sau R.420) + có dải phân cách | **60 km/h** |
| Trong khu dân cư (sau R.420) + không có dải phân cách | **50 km/h** |
| Ngoài khu dân cư (sau R.421 hoặc mặc định) + có dải phân cách | **90 km/h** |
| Ngoài khu dân cư (sau R.421 hoặc mặc định) + không có dải phân cách | **80 km/h** (mặc định) |

**Giả định:** xe là ô tô chở người đến 30 chỗ (trừ xe buýt) hoặc ô tô tải đến 3,5 tấn; thấy dải phân cách cứng thì coi là "đường đôi". Đường một chiều từ hai làn trở lên cũng được mức của đường đôi nhưng mô hình chưa nhận biết được. Không áp dụng cho đường cao tốc.

**Temporal Smoothing (Majority Voting):** Sử dụng cửa sổ lọc trượt (moving window). Tốc độ chỉ thay đổi khi một biển báo được tracker xác định ổn định tối thiểu $N$ frames, triệt tiêu hiện tượng sụt giảm tốc độ giả do biển báo bị che khuất hoặc lỗi nhận diện đơn lẻ.

---

## 🧪 Kiểm Thử

```bash
python tests/test_legal_speed_inference.py
```

Kiểm thử bao phủ bốn mức mặc định theo luật, việc ra/vào khu dân cư, biển P.127 ghi đè mức mặc định, và độ trễ của majority voting.

---

## 📌 Trạng Thái

- Toàn bộ pipeline (`train.py` → `evaluate.py` → `infer.py`) đã được chạy lại từ đầu trên dữ liệu mô phỏng (PyTorch 2.13, CPU) để xác nhận code chạy được.
- Repo **không kèm dữ liệu thật và checkpoint**, nên chưa có số liệu đánh giá nào tái lập được từ repo này. Các chỉ số in ra khi chạy trên dữ liệu mô phỏng không phản ánh chất lượng mô hình.
- Chưa có bản triển khai trên thiết bị nhúng (Jetson, TensorRT).

---

## 📄 Giấy Phép & Tuyên Bố Học Thuật

Dự án này được phát hành dưới **MIT License** cho mục đích học tập và phi thương mại.  
Các kỹ thuật tham khảo thuộc bản quyền của tác giả tương ứng (xem bảng Attribution ở trên).
