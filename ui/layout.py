"""原生 Gradio Blocks 布局；事件业务代码位于 callbacks.py。"""

import gradio as gr

from models import MODEL_NAMES, Predictor
from ui import presentation
from ui.callbacks import Controller, SessionData
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


def show_upload_page():
    return gr.update(visible=True), gr.update(visible=False)


def show_result_page(state: SessionData):
    # 只有成功生成结果才进入第二页；未上传或提取失败时留在上传页。
    ready = state.result is not None
    return gr.update(visible=not ready), gr.update(visible=ready)


def build_app(config: AppConfig, predictor: Predictor) -> gr.Blocks:
    controller = Controller(config, predictor)
    with gr.Blocks(
        title=f"{config.name} | {config.version}", analytics_enabled=False,
        fill_width=True, fill_height=True, delete_cache=(3600, 86400),
    ) as app:
        state = gr.State(SessionData())
        with gr.Column(elem_id="app-shell", min_width=0):
            gr.HTML(presentation.header(config), elem_id="app-heading")

            with gr.Column(elem_id="upload-page", min_width=0) as upload_page:
                preview = gr.Image(
                    label="上传预览", show_label=False, interactive=False,
                    format="png", height="100%", scale=1, buttons=[],
                    placeholder="上传遥感影像", elem_id="upload-preview",
                )
                info = gr.HTML(presentation.image_info(None), elem_id="image-info")
                with gr.Accordion("提取设置", open=False, elem_id="extract-settings"):
                    with gr.Row():
                        model = gr.Dropdown(
                            choices=list(MODEL_NAMES), value=config.default_model,
                            label="模型", interactive=True, min_width=160,
                        )
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
                    back_button = gr.Button("返回上传", scale=0, min_width=110, elem_id="back-button")
                    note = gr.HTML(presentation.result_note(None), scale=1, min_width=0)
                    save_button = gr.Button("保存结果", variant="primary", interactive=False, scale=0, min_width=110)
                    clear_button = gr.Button("清空结果", scale=0, min_width=110)

                with gr.Row(equal_height=True, elem_id="result-images"):
                    original = gr.Image(
                        label="原始遥感影像", interactive=False, format="png",
                        height="100%", buttons=["fullscreen"], min_width=0, elem_id="original-image",
                    )
                    mask = gr.Image(
                        label="分割掩码 · 模拟", image_mode="L", interactive=False, format="png",
                        height="100%", buttons=["fullscreen"], min_width=0, elem_id="mask-image",
                    )
                    overlay = gr.Image(
                        label="叠加结果 · 模拟", interactive=False, format="png",
                        height="100%", buttons=["fullscreen"], min_width=0, elem_id="overlay-image",
                    )

                with gr.Row(elem_id="visualization-controls"):
                    alpha = gr.Slider(0, 1, value=config.alpha, step=0.01, label="叠加透明度", scale=2, min_width=160)
                    color = gr.ColorPicker(value=config.water_color, label="水体颜色", scale=1, min_width=140)
                    boundary = gr.Checkbox(value=config.show_boundary, label="显示水体边界", scale=1, min_width=140)
                stats = gr.HTML(presentation.statistics(None), elem_id="result-statistics")
                downloads = gr.File(
                    label="下载结果", show_label=False, file_count="multiple",
                    interactive=False, visible=False, elem_id="saved-files",
                )

            # 保留可访问的状态播报，视觉上不再占用一整行说明区。
            status = gr.HTML(presentation.status("等待上传遥感影像。"), elem_id="operation-status")

        outputs = [state, preview, original, mask, overlay, info, stats, note, status, downloads, save_button]
        pages = [upload_page, result_page]
        event_options = dict(concurrency_id="water_workspace", concurrency_limit=1, show_progress="minimal")
        navigation_options = dict(api_name=False, **event_options)
        upload.upload(controller.upload, [upload], outputs, api_name="load_image", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        parameters = [upload, model, threshold, alpha, color, boundary, state]
        run_button.click(controller.run, parameters, outputs, api_name="extract", **event_options).then(
            show_result_page, [state], pages, **navigation_options,
        )
        gr.on([model.input, threshold.release], controller.invalidate, [state], outputs, api_name="invalidate", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        gr.on([alpha.release, color.change, boundary.change], controller.restyle, [upload, alpha, color, boundary, state], outputs, api_name="restyle", **event_options)
        save_button.click(controller.save, parameters, outputs, api_name="save_results", **event_options)
        clear_button.click(controller.clear, None, [upload, *outputs], api_name="clear", **event_options).then(
            show_upload_page, None, pages, **navigation_options,
        )
        back_button.click(show_upload_page, None, pages, **navigation_options)
    return app.queue(default_concurrency_limit=1, max_size=16)
