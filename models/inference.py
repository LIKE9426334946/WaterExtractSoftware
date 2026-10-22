"""后续 PyTorch 模型只需实现 Predictor.predict，无需依赖界面组件。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

MODEL_NAMES = ("U-Net", "U-Net++", "DeepLabv3+", "SegFormer")


@dataclass(frozen=True)
class Prediction:
    # H×W float32，范围 [0, 1]，与传入 RGB 图像尺寸完全相同。
    scores: np.ndarray
    backend_name: str
    is_simulation: bool


class Predictor(Protocol):
    def predict(self, image: np.ndarray, model_name: str) -> Prediction:
        """接收 H×W×3 uint8 RGB；返回原尺寸的水体分数/概率图。

        真实实现负责预处理、模型选择、推理、sigmoid/softmax 和尺寸恢复。
        模拟分数并非经过校准的水体概率。
        """
        ...


class SimulationPredictor:
    """仅按图像尺寸生成合成河道，完全不识别真实地物。"""

    def predict(self, image: np.ndarray, model_name: str) -> Prediction:
        if model_name not in MODEL_NAMES:
            raise ValueError("请选择受支持的模型。")
        height, width = image.shape[:2]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        center = 0.48 + 0.16 * np.sin(2 * np.pi * y + 0.5)
        river_width = 0.08 + 0.025 * np.cos(3 * np.pi * y)
        scores = 0.04 + 0.92 * np.exp(-0.5 * ((x - center) / river_width) ** 2)
        # 各模型选项使用同一模拟后端，不制造模型间的性能差异。
        return Prediction(scores.astype(np.float32), "合成河道模拟器", True)


def create_predictor(backend: str = "simulation") -> Predictor:
    if backend != "simulation":
        raise ValueError("本版本仅实现 simulation 后端，尚未接入真实模型。")
    return SimulationPredictor()
