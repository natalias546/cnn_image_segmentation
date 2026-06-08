import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VOCSegmentationDataset(Dataset):
    def __init__(self, root, split="train", image_size=(512, 512), augment=False):
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")
        self.image_size = image_size
        self.augment = augment

        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = Image.open(os.path.join(self.image_dir, f"{name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{name}.png"))

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        scale = random.uniform(1.0, 1.2)
        scaled_w = int(self.image_size[0] * scale)
        scaled_h = int(self.image_size[1] * scale)
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        mask = mask.resize((scaled_w, scaled_h), Image.NEAREST)
        i = random.randint(0, scaled_h - self.image_size[1])
        j = random.randint(0, scaled_w - self.image_size[0])
        image = TF.crop(image, i, j, self.image_size[1], self.image_size[0])
        mask = TF.crop(mask, i, j, self.image_size[1], self.image_size[0])

        return image, mask
