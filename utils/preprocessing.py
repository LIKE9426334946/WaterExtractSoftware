"""训练、验证和应用推理共用的 RGB 归一化。"""

import numpy as np
import torch


def image_to_tensor(image, mean, std):
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = (image - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
