"""界面文本与统计卡片；所有动态文本均进行 HTML 转义。"""

from html import escape

from utils.configuration import AppConfig
from utils.image_processing import LoadedImage
from utils.pipeline import ExtractionResult


def header(config: AppConfig) -> str:
    return f"""
    <header class="app-header">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
          <path d="M7 14c5-7 9 7 14 0s9 7 12 0M7 22c5-7 9 7 14 0s9 7 12 0M7 30c5-7 9 7 14 0s9 7 12 0" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="brand-copy"><h1>{escape(config.name)}</h1>
        <p>{escape(config.english_name)}</p></div>
      <span class="version">{escape(config.version)}</span>
    </header>"""


def status(message: str, tone: str = "neutral") -> str:
    return f'<div class="status-line {tone}" role="status"><span class="status-dot"></span>{escape(message)}</div>'


def image_info(source: LoadedImage | None) -> str:
    if source is None:
        return '<div class="image-meta empty-meta">等待上传影像 · 支持 JPG / JPEG / PNG<br>宽度、高度及像素分辨率将在此显示</div>'
    return f"""<div class="image-meta">
      <div class="filename">{escape(source.filename)}</div>
      <div class="meta-grid"><span>宽度 <b>{source.width:,} px</b></span>
      <span>高度 <b>{source.height:,} px</b></span></div>
      <div>像素分辨率：{source.width} × {source.height}（{source.width * source.height / 1e6:.2f} MP）</div>
      <div>地理分辨率：未知（未提供地理参考）</div>
    </div>"""


def statistics(result: ExtractionResult | None) -> str:
    if result is None:
        values = ["—"] * 5
    else:
        values = [
            f"{result.source.width} × {result.source.height}",
            f"{result.water_pixels:,}", f"{result.background_pixels:,}",
            f"{result.water_percent:.2f}%", f"{result.elapsed_ms:.2f} ms",
        ]
    labels = ["图像尺寸 / px", "水体像素数量", "非水体像素数量", "水体面积占比", "模拟推理耗时"]
    if result is not None and not result.is_simulation:
        labels[-1] = "推理耗时"
    cards = "".join(
        f'<div class="metric"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in zip(labels, values)
    )
    return f'<div class="metrics-grid">{cards}</div>'


def result_note(result: ExtractionResult | None) -> str:
    if result is None:
        return '<div class="result-note">尚未生成结果 · 上传影像后点击「开始提取」</div>'
    kind = "模拟结果" if result.is_simulation else "模型预测"
    return (
        f'<div class="result-note"><b>{kind}</b> · 模型选项：{escape(result.model_name)}'
        f' · 提取阈值：{result.threshold:.2f} · 后端：{escape(result.backend_name)}</div>'
    )
