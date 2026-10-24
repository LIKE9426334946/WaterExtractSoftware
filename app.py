"""WaterExtraction 启动入口：python app.py。"""

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import logging
from pathlib import Path

import gradio as gr
import uvicorn

from models import create_predictor
from ui.login import create_auth_app
from ui.layout import build_app, build_theme
from utils.authentication import AccountStore
from utils.configuration import load_config

ROOT = Path(__file__).resolve().parent


def create_app(config, predictor=None):
    accounts = AccountStore(config.users_file, config.session_hours * 3600)
    web, current_user = create_auth_app(config, accounts)
    blocks = build_app(config, predictor if predictor is not None else create_predictor(config))
    return gr.mount_gradio_app(
        web, blocks, path="/app", auth_dependency=current_user,
        theme=build_theme(),
        css_paths=ROOT / "ui" / "styles.css",
        max_file_size=config.max_upload_mb * 1024 * 1024,
        footer_links=[],
        blocked_paths=[str(config.users_file), str(config.users_file.with_suffix(".tmp"))],
        show_error=False,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(ROOT / "config.yaml")
    print(f"{config.name} {config.version} — http://{config.host}:{config.port}")
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
