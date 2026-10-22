"""WaterExtraction 启动入口：python app.py。"""

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import logging
from pathlib import Path

from models import create_predictor
from ui.layout import build_app, build_theme
from utils.configuration import load_config

ROOT = Path(__file__).resolve().parent


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(ROOT / "config.yaml")
    predictor = create_predictor(config)
    app = build_app(config, predictor)
    print(f"{config.name} {config.version} — 模型权重：{config.checkpoint}")
    app.launch(
        server_name=config.host,
        server_port=config.port,
        share=False,
        inbrowser=False,
        theme=build_theme(),
        css_paths=ROOT / "ui" / "styles.css",
        max_file_size=config.max_upload_mb * 1024 * 1024,
        footer_links=[],
        # 只允许 Gradio 按需缓存返回文件，不开放整个输出目录。
        show_error=False,
    )


if __name__ == "__main__":
    main()
