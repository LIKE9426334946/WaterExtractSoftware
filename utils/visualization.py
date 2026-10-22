"""只改变可视化，不修改二值掩码，也不重新运行推理。"""

import math

import numpy as np
from PIL import ImageColor


def unit_value(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须为 0～1 之间的数值。") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label}必须为 0～1 之间的数值。")
    return number


def parse_color(color: str) -> tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(color)
        if len(rgb) != 3:
            raise ValueError("不支持颜色中的独立透明通道。")
        return rgb
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("请选择有效的水体显示颜色，例如 #2563EB。") from exc


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """4 邻域内边界；图像边框外视为背景，不需要 OpenCV。"""
    water = mask.astype(bool)
    padded = np.pad(water, 1, constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return water & ~interior


def render_overlay(
    rgb: np.ndarray, mask: np.ndarray, alpha: float, color: str, show_boundary: bool = False
) -> np.ndarray:
    alpha = unit_value(alpha, "叠加透明度")
    tint = np.asarray(parse_color(color), dtype=np.float32)
    water = mask.astype(bool)
    if rgb.shape[:2] != water.shape:
        raise ValueError("叠加图与掩码的尺寸不一致。")
    result = rgb.copy()
    # 只在水体区域混色，背景像素严格保持不变。
    result[water] = np.rint((1 - alpha) * rgb[water] + alpha * tint).astype(np.uint8)
    if show_boundary:
        result[inner_boundary(water)] = tint.astype(np.uint8)
    return result
