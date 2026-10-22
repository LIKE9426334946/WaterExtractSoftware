"""按 CAID 官方 VOC 目录及 train/val 清单读取，不读取 test.txt。"""

import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


class CAIDDataset(Dataset):
    def __init__(self, config, split):
        data, train, model = config["dataset"], config["training"], config["model"]
        root = ROOT / data["root"]
        self.image_dir = root / data[f"{split}_images"]
        self.mask_dir = root / data[f"{split}_masks"]
        self.ids = (root / data[f"{split}_split"]).read_text(encoding="utf-8").split()
        self.image_suffix = data["image_suffix"]
        self.mask_suffix = data["mask_suffix"]
        self.training = split == "train"
        self.size = train["image_size"]
        self.scale_range = train["scale_range"]
        self.horizontal_flip_prob = train["horizontal_flip_prob"]
        self.vertical_flip_prob = train["vertical_flip_prob"]
        self.preprocess = smp.encoders.get_preprocessing_fn(
            model["encoder_name"], pretrained=model["encoder_weights"] or "imagenet"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        with Image.open(self.image_dir / f"{name}{self.image_suffix}") as source:
            image = source.convert("RGB")
        with Image.open(self.mask_dir / f"{name}{self.mask_suffix}") as source:
            # 官方定义：0/黑色为背景，非零值为水体；RGB 转换保留调色板颜色。
            # 同时支持官方单通道 0/1 及作者给出的黑/红调色板标注。
            water = np.any(np.asarray(source.convert("RGB")) != 0, axis=-1)
        mask = Image.fromarray(water.astype(np.uint8))

        if self.training:
            image, mask = self.augment(image, mask)

        # 只归一化 RGB；掩码始终保持 0/1，转为 BCE 所需的 float32。
        image = self.preprocess(np.asarray(image, dtype=np.float32)).astype(np.float32)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.array(mask, dtype=np.float32, copy=True)).unsqueeze(0)
        return image, mask

    def augment(self, image, mask):
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        size = (round(width * scale), round(height * scale))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)

        padding = (0, 0, max(0, self.size - size[0]), max(0, self.size - size[1]))
        image = ImageOps.expand(image, padding, fill=0)
        mask = ImageOps.expand(mask, padding, fill=0)
        left = random.randint(0, image.width - self.size)
        top = random.randint(0, image.height - self.size)
        box = (left, top, left + self.size, top + self.size)
        image, mask = image.crop(box), mask.crop(box)

        if random.random() < self.horizontal_flip_prob:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if random.random() < self.vertical_flip_prob:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        return image, mask
