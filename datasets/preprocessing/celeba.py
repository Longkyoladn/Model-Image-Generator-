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

def get_celeba_transforms(split="train", image_size=64, crop_size=140, norm_range="[-1, 1]", hflip=True):
    transform_list = [
        # Center crop phần trung tâm khuôn mặt (140x140 từ ảnh gốc 178x218)
        transforms.CenterCrop(crop_size),
        transforms.Resize((image_size, image_size))
    ]
    
    if split == "train" and hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    transform_list.append(transforms.ToTensor())
    
    if norm_range == "[-1, 1]":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    return transforms.Compose(transform_list)


class CelebADataset(Dataset):
    """
    Dataset linh hoạt cho CelebA: hỗ trợ cả thư mục giải nén từ Drive/Kaggle
    (img_align_celeba) và file phân vùng chính thức list_eval_partition.txt.
    """
    def __init__(self, root_dir="datasets/raw/celeba", split="train", transform=None, target_attr="Smiling"):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        
        # Mapping partition chuẩn CelebA: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        target_part = split_map.get(split, 0)
        
        # Tìm thư mục ảnh
        possible_img_dirs = [
            os.path.join(root_dir, "img_align_celeba"),
            os.path.join(root_dir, "celeba", "img_align_celeba"),
            root_dir
        ]
        self.img_dir = None
        for d in possible_img_dirs:
            if os.path.exists(d) and any(f.lower().endswith(('.jpg', '.png')) for f in os.listdir(d)[:5]):
                self.img_dir = d
                break
                
        self.samples = []
        self.attr_labels = []
        
        # Nếu có file phân vùng list_eval_partition.txt
        partition_file = os.path.join(root_dir, "list_eval_partition.txt")
        if not os.path.exists(partition_file):
            partition_file = os.path.join(root_dir, "celeba", "list_eval_partition.txt")
            
        attr_file = os.path.join(root_dir, "list_attr_celeba.txt")
        if not os.path.exists(attr_file):
            attr_file = os.path.join(root_dir, "celeba", "list_attr_celeba.txt")
            
        # Đọc nhãn thuộc tính nếu có
        attr_dict = {}
        target_attr_idx = -1
        if os.path.exists(attr_file):
            with open(attr_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) >= 2:
                    header = lines[1].split()
                    if target_attr in header:
                        target_attr_idx = header.index(target_attr)
                    for line in lines[2:]:
                        parts = line.split()
                        img_name = parts[0]
                        if target_attr_idx != -1 and len(parts) > target_attr_idx + 1:
                            val = int(parts[target_attr_idx + 1])
                            attr_dict[img_name] = 1 if val == 1 else 0

        if os.path.exists(partition_file) and self.img_dir:
            with open(partition_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_name, part_idx = parts[0], int(parts[1])
                        if part_idx == target_part:
                            full_path = os.path.join(self.img_dir, img_name)
                            if os.path.exists(full_path):
                                self.samples.append(full_path)
                                self.attr_labels.append(attr_dict.get(img_name, 0))
        elif self.img_dir:
            # Fallback nếu không có file txt: quét trực tiếp file ảnh và chia theo tỷ lệ
            all_files = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))])
            n_total = len(all_files)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            
            if split == "train":
                self.samples = all_files[:n_train]
            elif split == "val":
                self.samples = all_files[n_train:n_train + n_val]
            else:
                self.samples = all_files[n_train + n_val:]
            self.attr_labels = [0] * len(self.samples)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.attr_labels[idx] if idx < len(self.attr_labels) else 0
        return image, label


def load_celeba(root_dir="datasets/raw/celeba", split="train", image_size=64, crop_size=140, norm_range="[-1, 1]", target_attr="Smiling"):
    transform = get_celeba_transforms(
        split=split,
        image_size=image_size,
        crop_size=crop_size,
        norm_range=norm_range,
        hflip=(split == "train")
    )
    return CelebADataset(root_dir=root_dir, split=split, transform=transform, target_attr=target_attr)

