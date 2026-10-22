"""保留影像像素尺寸，校验真实格式，并处理 EXIF 方向及透明通道。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray
    path: str
    filename: str
    format: str

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]


def load_image(filepath: str | Path, max_upload_mb: int, max_pixels: int) -> LoadedImage:
    if not filepath:
        raise ValueError("请先上传 JPG、JPEG 或 PNG 影像。")
    path = Path(filepath)
    expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    suffix = path.suffix.lower()
    if suffix not in expected:
        raise ValueError("文件格式不支持，请选择 JPG、JPEG 或 PNG。")
    try:
        if path.stat().st_size > max_upload_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {max_upload_mb} MB，请选择更小的影像。")
        with Image.open(path) as source:
            image_format = source.format
            if image_format != expected[suffix]:
                raise ValueError("文件扩展名与真实图像格式不一致，请重新导出为 JPG 或 PNG。")
            if source.width * source.height > max_pixels:
                raise ValueError(f"图像超过 {max_pixels:,} 像素上限，请先裁剪后再上传。")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("仅支持单帧图像，请将动画导出为静态 PNG。")
            source.verify()
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                oriented = Image.alpha_composite(background, rgba)
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图像，文件可能已损坏或不是有效的 JPG/PNG。") from exc
    return LoadedImage(rgb, str(path), path.name, str(image_format))


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != shape:
        raise ValueError("推理接口输出尺寸与原始影像不一致。")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("推理接口必须输出有限的 [0, 1] 分数图。")
    return scores
