import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

INTERPOLATION_MAP = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
    "box": transforms.InterpolationMode.BOX,
    "hamming": transforms.InterpolationMode.HAMMING,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}

def get_celeba_transforms(
    split="train",
    image_size=64,
    crop_size=140,
    crop_type="center_crop",
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5
):
    transform_list = []
    interp_mode = INTERPOLATION_MAP.get(str(interpolation).lower(), transforms.InterpolationMode.BILINEAR)
    
    # 1. Crop khuôn mặt theo cấu hình YAML
    if crop_type == "center_crop" and crop_size:
        transform_list.append(transforms.CenterCrop(crop_size))
        
    # 2. Resize theo cấu hình YAML
    transform_list.append(transforms.Resize((image_size, image_size), interpolation=interp_mode))
    
    # 3. Augmentation theo cấu hình YAML
    should_flip = (split == "train") if hflip is None else hflip
    if should_flip and flip_prob is not None and flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
    # 4. Chuyển sang Tensor [0, 1]
    transform_list.append(transforms.ToTensor())
    
    # 5. Chuẩn hóa theo cấu hình YAML
    if norm_range == "[-1, 1]":
        norm_mean = mean if mean is not None else [0.5, 0.5, 0.5]
        norm_std = std if std is not None else [0.5, 0.5, 0.5]
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
    return transforms.Compose(transform_list)


def find_celeba_img_dir(root_dir):
    """
    Tìm kiếm thư mục ảnh CelebA một cách linh hoạt, hỗ trợ cấu hình giải nén lồng nhau phổ biến
    từ Kaggle (img_align_celeba/img_align_celeba/*.jpg) và Google Drive.
    """
    candidates = [
        os.path.join(root_dir, "img_align_celeba", "img_align_celeba"),
        os.path.join(root_dir, "celeba", "img_align_celeba", "img_align_celeba"),
        os.path.join(root_dir, "img_align_celeba"),
        os.path.join(root_dir, "celeba", "img_align_celeba"),
        root_dir
    ]
    for d in candidates:
        if os.path.exists(d) and os.path.isdir(d):
            try:
                files = os.listdir(d)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files[:20]):
                    return d
            except OSError:
                continue
                
    # Quét đệ quy nếu không nằm ở các vị trí mặc định
    if os.path.exists(root_dir):
        for root, dirs, files in os.walk(root_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files[:20]):
                return root
    return None


def find_celeba_meta_file(root_dir, filenames):
    """
    Tìm kiếm file metadata (list_eval_partition.txt/.csv hoặc list_attr_celeba.txt/.csv)
    trong root_dir hoặc các thư mục con. Hỗ trợ cả định dạng gốc .txt và Kaggle .csv.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
        
    for filename in filenames:
        candidates = [
            os.path.join(root_dir, filename),
            os.path.join(root_dir, "celeba", filename),
            os.path.join(root_dir, "img_align_celeba", filename),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                return c
                
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                if filename in files:
                    return os.path.join(root, filename)
    return None


class CelebADataset(Dataset):
    """
    Dataset linh hoạt cho CelebA:
    - Hỗ trợ phát hiện thư mục ảnh lồng nhau (Kaggle/Colab/Drive).
    - Hỗ trợ phân vùng chuẩn qua list_eval_partition.txt hoặc fallback phân chia tỷ lệ.
    - Bắt buộc kiểm tra file thuộc tính list_attr_celeba.txt khi chạy Conditional để tránh lỗi âm thầm toàn nhãn 0.
    """
    def __init__(self, root_dir="datasets/raw/celeba", split="train", transform=None, target_attr="Smiling", is_conditional=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        self.is_conditional = is_conditional
        
        # Mapping partition chuẩn CelebA: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        target_part = split_map.get(split, 0)
        
        # 1. Tìm thư mục ảnh
        self.img_dir = find_celeba_img_dir(root_dir)
        if self.img_dir is None:
            raise FileNotFoundError(
                f"[CelebADataset] Không tìm thấy thư mục chứa file ảnh (.jpg, .png) trong '{root_dir}'.\n"
                f"Kiểm tra lại cấu trúc giải nén trên Drive/Kaggle (ví dụ: 'img_align_celeba/img_align_celeba/')."
            )
            
        self.samples = []
        self.attr_labels = []
        
        # 2. Tìm file phân vùng và file thuộc tính (hỗ trợ cả file .txt gốc và file .csv từ Kaggle)
        partition_file = find_celeba_meta_file(root_dir, ["list_eval_partition.txt", "list_eval_partition.csv"])
        attr_file = find_celeba_meta_file(root_dir, ["list_attr_celeba.txt", "list_attr_celeba.csv"])
        
        # 3. Xử lý nhãn thuộc tính khi huấn luyện Conditional
        attr_dict = {}
        if self.is_conditional:
            if attr_file is None:
                raise FileNotFoundError(
                    f"[CelebADataset] Cấu hình yêu cầu huấn luyện Conditional (theo thuộc tính '{target_attr}'), "
                    f"nhưng không tìm thấy file 'list_attr_celeba.txt' hoặc 'list_attr_celeba.csv' trong '{root_dir}'.\n"
                    f"Vui lòng cung cấp file thuộc tính để mô hình có nhãn Smiling (0 và 1) chính xác!"
                )
            with open(attr_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) < 2:
                    raise ValueError(f"[CelebADataset] File '{attr_file}' không đúng định dạng thuộc tính CelebA.")
                
                # Hỗ trợ cả file CSV (dòng 0 là header) và file TXT (dòng 1 là header)
                if target_attr in lines[0].replace(",", " ").split():
                    header = lines[0].replace(",", " ").split()
                    data_lines = lines[1:]
                elif len(lines) >= 3 and target_attr in lines[1].replace(",", " ").split():
                    header = lines[1].replace(",", " ").split()
                    data_lines = lines[2:]
                else:
                    raise ValueError(
                        f"[CelebADataset] Thuộc tính '{target_attr}' không tồn tại trong '{attr_file}'."
                    )
                    
                target_attr_idx = header.index(target_attr)
                # Nếu header chứa cột tên ảnh (như image_id trong file CSV của Kaggle), target_attr_idx đã đúng vị trí cột.
                # Nếu header chỉ chứa danh sách thuộc tính (như file TXT gốc), cần +1 vì cột 0 của dữ liệu là tên ảnh.
                col_offset = 0 if header[0].lower() in ["image_id", "img_name", "image", "filename"] else 1
                for line in data_lines:
                    parts = line.replace(",", " ").split()
                    img_name = parts[0]
                    col_idx = target_attr_idx + col_offset
                    if len(parts) > col_idx:
                        val = int(parts[col_idx])
                        # CelebA quy ước 1 là có thuộc tính, -1 là không có thuộc tính
                        attr_dict[img_name] = 1 if val == 1 else 0
                        
        elif attr_file is not None:
            # Nếu chạy unconditional nhưng vẫn có file thuộc tính, đọc để tiện logging/đánh giá
            try:
                with open(attr_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if len(lines) >= 2:
                        if target_attr in lines[0].replace(",", " ").split():
                            header = lines[0].replace(",", " ").split()
                            data_lines = lines[1:]
                        elif len(lines) >= 3 and target_attr in lines[1].replace(",", " ").split():
                            header = lines[1].replace(",", " ").split()
                            data_lines = lines[2:]
                        else:
                            header, data_lines = [], []
                        if header and target_attr in header:
                            idx = header.index(target_attr)
                            col_offset = 0 if header[0].lower() in ["image_id", "img_name", "image", "filename"] else 1
                            for line in data_lines:
                                parts = line.replace(",", " ").split()
                                col_idx = idx + col_offset
                                if len(parts) > col_idx:
                                    attr_dict[parts[0]] = 1 if int(parts[col_idx]) == 1 else 0
            except Exception:
                pass

        # 4. Gom danh sách ảnh theo phân vùng split
        if partition_file and os.path.exists(partition_file):
            with open(partition_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.replace(",", " ").strip().split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        img_name, part_idx = parts[0], int(parts[1])
                        if part_idx == target_part:
                            full_path = os.path.join(self.img_dir, img_name)
                            if os.path.exists(full_path):
                                self.samples.append(full_path)
                                self.attr_labels.append(attr_dict.get(img_name, 0))
        else:
            # Fallback nếu không có file list_eval_partition.txt
            all_files = sorted([
                os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            n_total = len(all_files)
            if n_total == 0:
                raise FileNotFoundError(f"[CelebADataset] Thư mục '{self.img_dir}' không chứa file ảnh nào.")
                
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            
            if split == "train":
                self.samples = all_files[:n_train]
            elif split == "val":
                self.samples = all_files[n_train:n_train + n_val]
            else:
                self.samples = all_files[n_train + n_val:]
                
            self.attr_labels = [attr_dict.get(os.path.basename(p), 0) for p in self.samples]
            
        # Kiểm tra tính hợp lệ của nhãn khi chạy conditional
        if self.is_conditional:
            unique_labels = set(self.attr_labels)
            if len(unique_labels) < 2 and len(self.samples) > 100:
                raise RuntimeError(
                    f"[CelebADataset] Huấn luyện Conditional theo '{target_attr}' nhưng toàn bộ nhãn nạp được là {unique_labels}. "
                    f"Vui lòng kiểm tra lại sự khớp nhau giữa tên file ảnh trong '{self.img_dir}' và file '{attr_file}'."
                )
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.attr_labels[idx] if idx < len(self.attr_labels) else 0
        return image, label


def load_celeba(
    root_dir="datasets/raw/celeba",
    split="train",
    image_size=64,
    crop_size=140,
    crop_type="center_crop",
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5,
    target_attr="Smiling",
    is_conditional=False
):
    transform = get_celeba_transforms(
        split=split,
        image_size=image_size,
        crop_size=crop_size,
        crop_type=crop_type,
        interpolation=interpolation,
        norm_range=norm_range,
        mean=mean,
        std=std,
        hflip=hflip,
        flip_prob=flip_prob
    )
    return CelebADataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        target_attr=target_attr,
        is_conditional=is_conditional
    )
