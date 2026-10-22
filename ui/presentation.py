"""界面文本与统计卡片；所有动态文本均进行 HTML 转义。"""

from html import escape

from utils.configuration import AppConfig
from utils.image_processing import LoadedImage
from utils.pipeline import ExtractionResult


def header(config: AppConfig) -> str:
    return f"""
    <header class="app-header">
      <h1>{escape(config.name)}</h1>
      <span class="version">{escape(config.version)}</span>
    </header>"""


def status(message: str, tone: str = "neutral") -> str:
    return f'<div class="status-line {tone}" role="status"><span class="status-dot"></span>{escape(message)}</div>'


def image_info(source: LoadedImage | None) -> str:
    if source is None:
        return '<div class="image-meta">JPG / JPEG / PNG</div>'
    return (
        f'<div class="image-meta"><span class="filename">{escape(source.filename)}</span>'
        f'<span>{source.width} × {source.height} px · '
        f'{source.width * source.height / 1e6:.2f} MP</span></div>'
    )


def statistics(result: ExtractionResult | None) -> str:
    if result is None:
        values = ["—"] * 5
    else:
        values = [
            f"{result.source.width} × {result.source.height}",
            f"{result.water_pixels:,}", f"{result.background_pixels:,}",
            f"{result.water_percent:.2f}%", f"{result.elapsed_ms:.2f} ms",
        ]
    labels = ["图像尺寸 / px", "水体像素", "非水体像素", "水体占比", "推理耗时"]
    cards = "".join(
        f'<div class="metric"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in zip(labels, values)
    )
    return f'<div class="metrics-grid">{cards}</div>'


def result_note(result: ExtractionResult | None) -> str:
    if result is None:
        return ""
    return (
        f'<div class="result-note"><b>提取完成</b><span>阈值 {result.threshold:.2f}</span></div>'
    )
