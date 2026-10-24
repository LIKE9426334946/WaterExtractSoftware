"""历史列表与只读详情，不改变当前提取会话。"""

import logging
from pathlib import Path

import gradio as gr

from ui import presentation
from utils.history import SavedResult, list_results

logger = logging.getLogger(__name__)


class HistoryController:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def refresh(self):
        try:
            records = list_results(self.output_dir)
        except (OSError, ValueError):
            logger.exception("Reading saved results failed")
            raise gr.Error("无法读取保存结果，请检查输出目录及 history.json。")
        thumbnails = [(str(record.original or record.overlay), f"编号 {record.number}") for record in records]
        message = f"共 {len(records)} 条保存结果 · 点击编号查看" if records else "暂无保存结果，请先进行提取并点击“保存结果”。"
        return records, gr.update(value=thumbnails, selected_index=None), presentation.status(message)

    def open(self, records: list[SavedResult], event: gr.SelectData):
        if not isinstance(event.index, int) or not 0 <= event.index < len(records):
            raise gr.Error("请刷新列表后重新选择。")
        record = records[event.index]
        if not record.mask.is_file() or not record.overlay.is_file():
            raise gr.Error("该结果文件已被移除，请返回列表刷新。")
        return (
            str(record.original) if record.original else None,
            str(record.mask), str(record.overlay),
            presentation.saved_statistics(record.metadata),
            presentation.saved_note(record),
            record.downloads,
        )
