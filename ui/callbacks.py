"""事件适配：每个浏览器会话独立保存状态，错误时及时清除过期结果。"""

import logging
from dataclasses import dataclass, replace

import gradio as gr

from models import Predictor
from ui import presentation
from utils.configuration import AppConfig
from utils.exporting import save_result
from utils.image_processing import LoadedImage, load_image
from utils.pipeline import ExtractionResult, extract, redraw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionData:
    source: LoadedImage | None = None
    result: ExtractionResult | None = None


class Controller:
    def __init__(self, config: AppConfig, predictor: Predictor):
        self.config = config
        self.predictor = predictor

    @staticmethod
    def view(state: SessionData, message: str, *, tone: str = "neutral", downloads=None, only=None):
        """统一输出顺序与 layout.py 对应；重绘时跳过未改变的大图。"""
        source, result = state.source, state.result
        image = source.rgb if source else None
        values = {
            "state": state,
            "preview": image,
            "original": image,
            "mask": result.mask if result else None,
            "overlay": result.overlay if result else None,
            "info": presentation.image_info(source),
            "stats": presentation.statistics(result),
            "note": presentation.result_note(result),
            "status": presentation.status(message, tone),
            "downloads": gr.update(value=downloads, visible=bool(downloads)),
            "save": gr.update(interactive=result is not None),
        }
        return tuple(value if only is None or key in only else gr.skip() for key, value in values.items())

    def upload(self, filepath):
        if not filepath:
            return self.view(SessionData(), "等待上传遥感影像。")
        try:
            source = load_image(filepath, self.config.max_upload_mb, self.config.max_pixels)
            return self.view(SessionData(source), "影像已就绪，点击「开始提取」进行预测。", tone="ready")
        except ValueError as exc:
            gr.Warning(str(exc))
            return self.view(SessionData(), str(exc), tone="error")

    @staticmethod
    def require_source(filepath, state: SessionData) -> LoadedImage:
        if not filepath or state.source is None:
            raise ValueError("请先上传有效的 JPG、JPEG 或 PNG 影像。")
        if str(filepath) != state.source.path:
            raise ValueError("上传影像尚未就绪，请等待预览完成后重新提取。")
        return state.source

    def run(self, filepath, threshold, alpha, color, boundary, state: SessionData):
        try:
            source = self.require_source(filepath, state)
            result = extract(source, self.predictor, threshold, alpha, color, boundary)
            return self.view(SessionData(source, result), "水体提取完成。", tone="ready")
        except ValueError as exc:
            gr.Warning(str(exc))
            return self.view(replace(state, result=None), str(exc), tone="error")
        except Exception:
            logger.exception("Extraction failed")
            gr.Warning("提取失败，请检查终端日志后重试。")
            return self.view(replace(state, result=None), "提取失败，未生成结果。", tone="error")

    def invalidate(self, state: SessionData):
        message = "提取参数已变更，请点击「开始提取」重新生成结果。" if state.source else "请先上传遥感影像。"
        return self.view(replace(state, result=None), message, only={"state", "mask", "overlay", "stats", "note", "status", "downloads", "save"})

    def restyle(self, filepath, alpha, color, boundary, state: SessionData):
        only = {"state", "overlay", "status", "downloads", "save"}
        if state.result is None:
            return self.view(state, "可视化设置已更新，提取后生效。", only=only)
        try:
            self.require_source(filepath, state)
            result = redraw(state.result, alpha, color, boundary)
            return self.view(replace(state, result=result), "可视化已更新；可点击「保存结果」导出当前版本。", tone="ready", only=only)
        except ValueError as exc:
            gr.Warning(str(exc))
            return self.view(state, str(exc), tone="error", only={"status", "downloads"})

    def save(self, filepath, threshold, alpha, color, boundary, state: SessionData):
        try:
            self.require_source(filepath, state)
            if state.result is None:
                raise ValueError("尚无可保存的结果，请先点击「开始提取」。")
            if state.result.threshold != float(threshold):
                raise ValueError("阈值已改变，请重新提取后再保存。")
            # 从当前控件读取样式，避免滑块更新与保存之间出现过期导出。
            result = redraw(state.result, alpha, color, boundary)
            files = save_result(result, self.config.output_dir)
            return self.view(replace(state, result=result), "保存成功：掩码、叠加图和统计已写入输出目录；可在下方下载。", tone="ready", downloads=files, only={"state", "overlay", "status", "downloads", "save"})
        except ValueError as exc:
            gr.Warning(str(exc))
            return self.view(state, str(exc), tone="error", only={"status", "downloads"})
        except OSError:
            logger.exception("Saving results failed")
            gr.Warning("保存失败，请检查输出目录的写入权限与磁盘空间。")
            return self.view(state, "保存失败，请检查输出目录权限与磁盘空间。", tone="error", only={"status", "downloads"})

    def clear(self):
        # 只清理本会话；用户已经保存的文件不会被删除。
        return (None, *self.view(SessionData(), "已清空影像、提取结果和统计信息。"))
