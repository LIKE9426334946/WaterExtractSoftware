"""提取业务流程：推理、阈值化、统计与重绘；可脱离 Gradio 使用。"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from models import MODEL_NAMES, Predictor
from utils.image_processing import LoadedImage, validate_scores
from utils.visualization import render_overlay, unit_value


@dataclass(frozen=True)
class ExtractionResult:
    source: LoadedImage
    mask: np.ndarray  # H×W uint8，值为 0 或 255。
    overlay: np.ndarray
    model_name: str
    backend_name: str
    is_simulation: bool
    threshold: float
    alpha: float
    color: str
    show_boundary: bool
    elapsed_ms: float
    water_pixels: int

    @property
    def total_pixels(self) -> int:
        return self.source.width * self.source.height

    @property
    def background_pixels(self) -> int:
        return self.total_pixels - self.water_pixels

    @property
    def water_percent(self) -> float:
        return self.water_pixels / self.total_pixels * 100


def extract(
    source: LoadedImage, predictor: Predictor, model_name: str, threshold: float,
    alpha: float, color: str, show_boundary: bool,
) -> ExtractionResult:
    threshold = unit_value(threshold, "预测阈值")
    if model_name not in MODEL_NAMES:
        raise ValueError("请选择受支持的模型。")
    start = perf_counter()
    prediction = predictor.predict(source.rgb, model_name)
    scores = validate_scores(prediction.scores, source.rgb.shape[:2])
    mask = (scores >= threshold).astype(np.uint8) * 255
    # 耗时只包括 predict + 校验 + 阈值化，不包括图像显示、上传和保存。
    elapsed_ms = (perf_counter() - start) * 1000
    overlay = render_overlay(source.rgb, mask, alpha, color, show_boundary)
    return ExtractionResult(
        source, mask, overlay, model_name, prediction.backend_name,
        prediction.is_simulation, threshold, float(alpha), color, bool(show_boundary),
        elapsed_ms, int(np.count_nonzero(mask)),
    )


def redraw(result: ExtractionResult, alpha: float, color: str, boundary: bool) -> ExtractionResult:
    overlay = render_overlay(result.source.rgb, result.mask, alpha, color, boundary)
    return replace(result, overlay=overlay, alpha=float(alpha), color=color, show_boundary=bool(boundary))
