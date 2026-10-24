"""原生 Gradio Blocks 布局；事件业务代码位于 callbacks.py。"""

import gradio as gr

from models import Predictor
from ui import presentation
from ui.callbacks import Controller, SessionData
from ui.history import HistoryController
from utils.configuration import AppConfig


def build_theme():
    theme = gr.themes.Soft(
        primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
        font=["Microsoft YaHei", "PingFang SC", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"], radius_size="lg",
    )
    # 使用本地字体，并保持浅色科研工作台风格。
    palette = {
        "body_background_fill": "#f4f7fb", "body_text_color": "#1e293b",
        "body_text_color_subdued": "#64748b", "background_fill_primary": "#ffffff",
        "background_fill_secondary": "#f8fafc", "block_background_fill": "#ffffff",
        "block_border_color": "#e2e8f0", "block_label_text_color": "#475569",
        "block_label_background_fill": "#ffffff", "input_background_fill": "#f8fafc",
        "input_border_color": "#e2e8f0", "input_placeholder_color": "#94a3b8",
        "button_primary_background_fill": "#2563eb", "button_primary_text_color": "#ffffff",
        "button_primary_background_fill_hover": "#1d4ed8",
        "button_secondary_background_fill": "#ffffff", "button_secondary_text_color": "#334155",
        "button_secondary_background_fill_hover": "#eff6ff", "border_color_primary": "#e2e8f0",
    }
    return theme.set(**palette, **{f"{key}_dark": value for key, value in palette.items()})


PAGE_NAMES = ("home", "upload", "result", "history", "saved")


def show_page(name: str):
    return tuple(gr.update(visible=page == name) for page in PAGE_NAMES)


def show_upload_page():
    return show_page("upload")


def show_result_page(state: SessionData):
    # 只有成功生成结果才进入结果页；未上传或提取失败时留在上传页。
    ready = state.result is not None
    return show_page("result" if ready else "upload")


def result_images(prefix: str = ""):
    with gr.Row(equal_height=True, elem_id=f"{prefix}result-images", elem_classes="result-images"):
        images = []
        for name, label in (("original", "原始遥感影像"), ("mask", "分割掩码"), ("overlay", "叠加结果")):
            images.append(gr.Image(
                label=label, image_mode="L" if name == "mask" else "RGB",
                interactive=False, format="png", height="100%", buttons=["fullscreen"],
                min_width=0, elem_id=f"{prefix}{name}-image", elem_classes="result-image",
            ))
    return images


def build_app(config: AppConfig, predictor: Predictor) -> gr.Blocks:
    controller = Controller(config, predictor)
    history = HistoryController(config.output_dir)
    with gr.Blocks(
        title=f"{config.name} | {config.version}", analytics_enabled=False,
        fill_width=True, fill_height=True, delete_cache=(3600, 86400),
    ) as app:
        state = gr.State(SessionData())
        history_records = gr.State([])
        with gr.Column(elem_id="app-shell", min_width=0):
            gr.HTML(presentation.header(config), elem_id="app-heading")

            with gr.Column(elem_id="home-page", min_width=0) as home_page:
                with gr.Row(elem_id="home-options", equal_height=True):
                    with gr.Column(elem_classes="home-card", min_width=0):
                        gr.HTML('<div class="home-icon" aria-hidden="true"><svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10 24V12h12m20 0h12v12m0 16v12H42m-20 0H10V40"/><path d="M18 32c5-8 9 8 14 0s9 8 14 0M18 41c5-8 9 8 14 0s9 8 14 0"/></svg></div>')
                        start_button = gr.Button("开始进行提取", variant="primary", size="lg", elem_id="start-extraction")
                    with gr.Column(elem_classes="home-card", min_width=0):
                        gr.HTML('<div class="home-icon" aria-hidden="true"><svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="12" y="12" width="40" height="40" rx="5"/><circle cx="24" cy="25" r="4"/><path d="m13 45 12-11 8 7 9-14 10 18"/></svg></div>')
                        history_button = gr.Button("查看保存结果", size="lg", elem_id="view-history")

            with gr.Column(visible=False, elem_id="upload-page", min_width=0) as upload_page:
                with gr.Row(elem_classes="page-toolbar"):
                    upload_home = gr.Button("返回首页", scale=0, min_width=100)
                    gr.HTML('<h2 class="page-title">开始进行提取</h2>')
                preview = gr.Image(
                    label="上传预览", show_label=False, interactive=False,
                    format="png", height="100%", scale=1, buttons=[],
                    placeholder="上传遥感影像", elem_id="upload-preview",
                )
                info = gr.HTML(presentation.image_info(None), elem_id="image-info")
                with gr.Accordion("提取设置", open=False, elem_id="extract-settings"):
                    with gr.Row():
                        threshold = gr.Slider(
                            0, 1, value=config.threshold, step=0.01,
                            label="水体预测阈值", min_width=160,
                        )
                with gr.Row(elem_id="upload-actions"):
                    upload = gr.UploadButton(
                        "上传图像", type="filepath", size="lg", min_width=140,
                        file_types=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
                        elem_id="image-upload",
                    )
                    run_button = gr.Button(
                        "开始提取", variant="primary", size="lg",
                        min_width=140, elem_id="run-button",
                    )

            with gr.Column(visible=False, elem_id="result-page", min_width=0) as result_page:
                with gr.Row(elem_id="result-toolbar"):
                    result_home = gr.Button("返回首页", scale=0, min_width=100)
                    back_button = gr.Button("返回上传", scale=0, min_width=110, elem_id="back-button")
                    note = gr.HTML(presentation.result_note(None), scale=1, min_width=0)
                    save_button = gr.Button("保存结果", variant="primary", interactive=False, scale=0, min_width=110)
                    clear_button = gr.Button("清空结果", scale=0, min_width=110)

                original, mask, overlay = result_images()

                with gr.Row(elem_id="visualization-controls"):
                    alpha = gr.Slider(0, 1, value=config.alpha, step=0.01, label="叠加透明度", scale=2, min_width=160)
                    color = gr.ColorPicker(value=config.water_color, label="水体颜色", scale=1, min_width=140)
                    boundary = gr.Checkbox(value=config.show_boundary, label="显示水体边界", scale=1, min_width=140)
                stats = gr.HTML(presentation.statistics(None), elem_id="result-statistics")
                downloads = gr.File(
                    label="下载结果", show_label=False, file_count="multiple",
                    interactive=False, visible=False, elem_id="saved-files", elem_classes="download-files",
                )

            with gr.Column(visible=False, elem_id="history-page", min_width=0) as history_page:
                with gr.Row(elem_classes="page-toolbar"):
                    history_home = gr.Button("返回首页", scale=0, min_width=100)
                    gr.HTML('<h2 class="page-title">查看保存结果</h2>')
                    refresh_history = gr.Button("刷新列表", scale=0, min_width=100)
                history_summary = gr.HTML(elem_id="history-summary")
                gallery = gr.Gallery(
                    value=[], label="保存记录", show_label=False, interactive=False,
                    columns=4, rows=2, height="100%", object_fit="contain",
                    allow_preview=False, buttons=[], elem_id="history-gallery",
                )

            with gr.Column(visible=False, elem_id="saved-page", min_width=0) as saved_page:
                with gr.Row(elem_classes="page-toolbar"):
                    saved_home = gr.Button("返回首页", scale=0, min_width=100)
                    back_history = gr.Button("返回列表", scale=0, min_width=100)
                    saved_note = gr.HTML(elem_id="saved-note")
                saved_original, saved_mask, saved_overlay = result_images("saved-")
                saved_stats = gr.HTML(elem_id="saved-statistics")
                saved_downloads = gr.File(
                    label="下载结果", show_label=False, file_count="multiple",
                    interactive=False, elem_id="history-downloads", elem_classes="download-files",
                )

            # 保留可访问的状态播报，视觉上不再占用一整行说明区。
            status = gr.HTML(presentation.status("等待上传遥感影像。"), elem_id="operation-status")

        outputs = [state, preview, original, mask, overlay, info, stats, note, status, downloads, save_button]
        pages = [home_page, upload_page, result_page, history_page, saved_page]
        event_options = dict(concurrency_id="water_workspace", concurrency_limit=1, show_progress="minimal")
        navigation_options = dict(api_name=False, **event_options)
        start_button.click(show_upload_page, None, pages, **navigation_options)
        gr.on([button.click for button in (upload_home, result_home, history_home, saved_home)],
              lambda: show_page("home"), None, pages, **navigation_options)
        gr.on([history_button.click, refresh_history.click, back_history.click],
              history.refresh, None, [history_records, gallery, history_summary],
              api_name="list_saved_results", **event_options).success(
            lambda: show_page("history"), None, pages, **navigation_options,
        )
        gallery.select(
            history.open, [history_records],
            [saved_original, saved_mask, saved_overlay, saved_stats, saved_note, saved_downloads],
            **navigation_options,
        ).success(lambda: show_page("saved"), None, pages, **navigation_options)
        upload.upload(controller.upload, [upload], outputs, api_name="load_image", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        parameters = [upload, threshold, alpha, color, boundary, state]
        run_button.click(controller.run, parameters, outputs, api_name="extract", **event_options).then(
            show_result_page, [state], pages, **navigation_options,
        )
        threshold.release(controller.invalidate, [state], outputs, api_name="invalidate", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        gr.on([alpha.release, color.change, boundary.change], controller.restyle, [upload, alpha, color, boundary, state], outputs, api_name="restyle", **event_options)
        save_button.click(controller.save, parameters, outputs, api_name="save_results", **event_options)
        clear_button.click(controller.clear, None, [upload, *outputs], api_name="clear", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        back_button.click(show_upload_page, None, pages, **navigation_options)
    return app.queue(default_concurrency_limit=1, max_size=16)
