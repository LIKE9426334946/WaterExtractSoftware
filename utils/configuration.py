"""加载配置；路径不依赖启动命令所在的工作目录。"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from models import MODEL_NAMES
from utils.visualization import parse_color, unit_value


@dataclass(frozen=True)
class AppConfig:
    name: str
    english_name: str
    version: str
    host: str
    port: int
    backend: str
    default_model: str
    threshold: float
    alpha: float
    water_color: str
    show_boundary: bool
    output_dir: Path
    max_upload_mb: int
    max_pixels: int


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    try:
        app, server = data["app"], data["server"]
        inference, vis, files = data["inference"], data["visualization"], data["files"]
        config = AppConfig(
            name=str(app["name"]), english_name=str(app["english_name"]),
            version=str(app["version"]), host=str(server["host"]),
            port=int(server["port"]), backend=str(inference["backend"]),
            default_model=str(inference["default_model"]),
            threshold=unit_value(inference["threshold"], "预测阈值"),
            alpha=unit_value(vis["alpha"], "叠加透明度"),
            water_color=str(vis["water_color"]), show_boundary=vis["show_boundary"],
            output_dir=(path.parent / files["output_dir"]).resolve(),
            max_upload_mb=int(files["max_upload_mb"]), max_pixels=int(files["max_pixels"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config.yaml 配置无效：{exc}") from exc
    parse_color(config.water_color)
    if not 1 <= config.port <= 65535:
        raise ValueError("server.port 必须在 1～65535 之间。")
    if config.default_model not in MODEL_NAMES:
        raise ValueError("inference.default_model 不在支持的模型列表中。")
    if not isinstance(config.show_boundary, bool):
        raise ValueError("visualization.show_boundary 必须为 true 或 false。")
    if config.max_upload_mb <= 0 or config.max_pixels <= 0:
        raise ValueError("文件大小和像素上限必须为正数。")
    return config
