"""
src/dataset.py – Dataset và DataLoader cho Dự Án ADAS Việt Nam
================================================================
Đồ Án Phân Tích Dữ Liệu – Nhóm 6 (GR6) – Kỳ 4

Mô tả:
    ADASDataset: PyTorch Dataset đọc ảnh (.jpg/.png), nhãn YOLO (.txt)
    và mặt nạ phân vùng (.png). Mã hóa nhãn sang dạng lưới (ô vuông)
    phù hợp với YOLO-style detection head.

Lưu ý: Đây là code tự viết, không dùng thư viện dataset có sẵn.
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ADASDataset(Dataset):
    def __init__(self, img_dir, label_dir, mask_dir, img_size=640, grid_size=40, num_classes_det=7, transform=None):
        """
        img_dir: Directory containing images
        label_dir: Directory containing YOLO detection labels (.txt)
        mask_dir: Directory containing segmentation masks (.png)
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes_det = num_classes_det
        self.transform = transform
        
        # Get list of image names (without extension)
        self.image_files = []
        if os.path.exists(img_dir):
            self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            self.image_files.sort()
        else:
            print(f"Warning: Image directory {img_dir} does not exist.")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        
        # 1. LOAD IMAGE
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image file: {img_path}")
        h_orig, w_orig = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 2. LOAD SEGMENTATION MASK
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Mask file {mask_path} could not be read. Using empty mask.")
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        # 3. LOAD DETECTION LABELS
        bboxes = [] # List of [class_id, x_center, y_center, w, h]
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # Check boundaries of class_id
                            if class_id >= self.num_classes_det:
                                continue
                            coords = [float(x) for x in parts[1:5]]
                            bboxes.append([class_id] + coords)
            except Exception as e:
                print(f"Warning: Failed to read label file {label_path}: {e}")
                
        bboxes = np.array(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 5), dtype=np.float32)
        
        # Encode labels into grid format: [S, S, 5 + C]
        # target_grid[..., 0]: objectness confidence
        # target_grid[..., 1:3]: x, y relative to grid cell (0 to 1)
        # target_grid[..., 3:5]: w, h relative to full image (0 to 1)
        # target_grid[..., 5:]: one-hot class probabilities
        encoded_det = np.zeros((self.grid_size, self.grid_size, 5 + self.num_classes_det), dtype=np.float32)
        
        for box in bboxes:
            class_id, x_center, y_center, w, h = box
            
            # Find grid coordinates
            cell_x = int(x_center * self.grid_size)
            cell_y = int(y_center * self.grid_size)
            
            # Clamp to grid bounds
            cell_x = max(0, min(self.grid_size - 1, cell_x))
            cell_y = max(0, min(self.grid_size - 1, cell_y))
            
            # Cell relative coordinate
            x_cell = (x_center * self.grid_size) - cell_x
            y_cell = (y_center * self.grid_size) - cell_y
            
            # Store in target grid
            encoded_det[cell_y, cell_x, 0] = 1.0
            encoded_det[cell_y, cell_x, 1] = x_cell
            encoded_det[cell_y, cell_x, 2] = y_cell
            encoded_det[cell_y, cell_x, 3] = w
            encoded_det[cell_y, cell_x, 4] = h
            encoded_det[cell_y, cell_x, 5 + int(class_id)] = 1.0
            
        # Transform image
        # Normalization and to Tensor
        image = image.astype(np.float32) / 255.0
        # Transpose to [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        mask = torch.from_numpy(mask.astype(np.int64))
        encoded_det = torch.from_numpy(encoded_det)
        bboxes = torch.from_numpy(bboxes)
        
        return image, mask, encoded_det, bboxes

def collate_fn(batch):
    images, masks, encoded_dets, raw_boxes = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    encoded_dets = torch.stack(encoded_dets, dim=0)
    # raw_boxes can be kept as a tuple of tensors to accommodate varying count of boxes per image
    return images, masks, encoded_dets, raw_boxes

def get_dataloader(img_dir, label_dir, mask_dir, img_size=640, grid_size=40, 
                   num_classes_det=7, batch_size=8, shuffle=True, num_workers=4):
    dataset = ADASDataset(img_dir, label_dir, mask_dir, img_size, grid_size, num_classes_det)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
