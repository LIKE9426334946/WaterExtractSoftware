"""加载训练得到的融合网络权重，以重叠分块方式返回原尺寸水体概率图。"""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np
import torch

from .model_factory import create_model
from utils.preprocessing import image_to_tensor

if TYPE_CHECKING:
    from utils.configuration import AppConfig


@dataclass(frozen=True)
class Prediction:
    scores: np.ndarray  # H×W float32，范围 [0, 1]，与输入 RGB 图像同尺寸。
    model_name: str


class Predictor(Protocol):
    def predict(self, image: np.ndarray) -> Prediction:
        """接收 H×W×3 uint8 RGB，返回原尺寸的水体概率图。"""
        ...


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    """最后一块贴齐影像边缘，确保每个像素都被覆盖。"""
    last = max(0, length - size)
    starts = list(range(0, last + 1, size - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


class TorchPredictor:
    def __init__(self, config: "AppConfig"):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.amp = config.amp and self.device.type == "cuda"
        self.model = None

    def load_model(self):
        # 首次提取时加载，此后复用；缺失或不匹配的权重不会产生随机预测结果。
        if self.model is not None:
            return
        if not self.config.checkpoint.is_file():
            raise ValueError(
                f"未找到模型权重：{self.config.checkpoint}。"
                "请先运行 python train.py，或修改 config.yaml 中的 inference.checkpoint。"
            )
        weights = torch.load(self.config.checkpoint, map_location="cpu", weights_only=True)
        model = create_model(self.config.model)
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise ValueError("权重与模型结构不匹配，请使用训练时的 model 配置和对应的 best.pth。") from exc
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        self.load_model()
        height, width = image.shape[:2]
        size, overlap = self.config.tile_size, self.config.tile_overlap
        scores = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros_like(scores)
        for top in tile_starts(height, size, overlap):
            for left in tile_starts(width, size, overlap):
                tile = image[top:top + size, left:left + size]
                tile_height, tile_width = tile.shape[:2]
                # 小图复制边缘填充；保持原始像素比例，不将影像拉伸成正方形。
                tile = np.pad(tile, ((0, size - tile_height), (0, size - tile_width), (0, 0)), mode="edge")
                tensor = image_to_tensor(tile, self.config.image_mean, self.config.image_std)
                tensor = tensor.unsqueeze(0).to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    logits = self.model(tensor)
                prob = torch.sigmoid(logits.float())[0, 0, :tile_height, :tile_width].cpu().numpy()
                scores[top:top + tile_height, left:left + tile_width] += prob
                counts[top:top + tile_height, left:left + tile_width] += 1
        scores /= counts
        return Prediction(scores, type(self.model).__name__)


def create_predictor(config: "AppConfig") -> Predictor:
    return TorchPredictor(config)
