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


def build_app(config: AppConfig, predictor: Predictor) -> gr.Blocks:
    controller = Controller(config, predictor)
    with gr.Blocks(title=f"{config.name} | {config.version}", analytics_enabled=False, fill_width=True, delete_cache=(3600, 86400)) as app:
        state = gr.State(SessionData())
        gr.HTML(presentation.header(config))
        gr.HTML('<div class="simulation-banner"><b>模拟演示模式</b><span>未加载真实模型 · 当前结果仅用于软件界面测试，不代表真实水体分割。</span></div>')

        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=1, min_width=300, elem_id="operation-panel"):
                gr.HTML('<div class="section-title"><span class="step-number">01</span><h2>影像与参数</h2></div>')
                upload = gr.File(label="上传遥感影像", file_types=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"], type="filepath", height=100, elem_id="image-upload")
                preview = gr.Image(label="上传预览", interactive=False, format="png", height=150, buttons=[], placeholder="上传后预览影像", elem_id="upload-preview")
                info = gr.HTML(presentation.image_info(None))
                model = gr.Dropdown(choices=list(MODEL_NAMES), value=config.default_model, label="模型选择", info="仅选择模型名称，当前均使用同一模拟后端。", interactive=True)
                threshold = gr.Slider(0, 1, value=config.threshold, step=0.01, label="水体预测阈值", info="修改后需重新提取。")
                alpha = gr.Slider(0, 1, value=config.alpha, step=0.01, label="叠加透明度", info="0：完全透明 · 1：完全着色")
                with gr.Row():
                    color = gr.ColorPicker(value=config.water_color, label="水体显示颜色", min_width=120)
                    boundary = gr.Checkbox(value=config.show_boundary, label="显示水体边界", info="实色描绘，不改变掩码。", min_width=120)
                run_button = gr.Button("开始提取", variant="primary", size="lg", elem_id="run-button")
                with gr.Row():
                    clear_button = gr.Button("清空结果", size="sm")
                    save_button = gr.Button("保存结果", size="sm", interactive=False)
                gr.HTML(f'<div class="panel-footnote">单张 ≤ {config.max_upload_mb} MB · ≤ {config.max_pixels / 1e6:g} MP<br>保持原始宽高比例与像素尺寸</div>')

            with gr.Column(scale=3, min_width=560, elem_id="result-panel"):
                gr.HTML('<div class="section-title"><span class="step-number">02</span><h2>结果工作区</h2><span class="section-extra">影像对照 / 像素级可视化</span></div>')
                note = gr.HTML(presentation.result_note(None))
                with gr.Row(equal_height=True, elem_id="result-images"):
                    original = gr.Image(label="原始遥感影像", interactive=False, format="png", height=320, buttons=["fullscreen"], placeholder="原始影像", min_width=200, elem_id="original-image")
                    mask = gr.Image(label="水体二值分割掩码 · 模拟", image_mode="L", interactive=False, format="png", height=320, buttons=["fullscreen"], placeholder="等待模拟提取", min_width=200, elem_id="mask-image")
                    overlay = gr.Image(label="水体提取叠加结果 · 模拟", interactive=False, format="png", height=320, buttons=["fullscreen"], placeholder="等待模拟提取", min_width=200, elem_id="overlay-image")
                gr.HTML('<div class="legend"><span><i class="legend-square white"></i>白色：水体</span><span><i class="legend-square black"></i>黑色：背景</span><span>叠加结果支持透明度、颜色与边界显示</span></div>')
                gr.HTML('<div class="subheading"><h3>结果统计</h3><span>基于当前原尺寸掩码计算</span></div>')
                stats = gr.HTML(presentation.statistics(None), elem_id="result-statistics")
                gr.HTML('<div class="statistics-note">水体面积占比 = 水体像素数 ÷ 总像素数。此处不计算真实地理面积；模拟耗时不代表模型性能。</div>')
                status = gr.HTML(presentation.status("等待上传遥感影像。"), elem_id="operation-status")
                downloads = gr.File(label="下载已保存结果（PNG / ZIP）", file_count="multiple", interactive=False, visible=False, elem_id="saved-files")
                gr.HTML('<div class="quick-guide"><h3>使用流程</h3><p><b>1</b> 上传影像 <span>→</span> <b>2</b> 设置阈值并提取 <span>→</span> <b>3</b> 调整显示效果 <span>→</span> <b>4</b> 保存与下载</p><div>模拟器仅根据影像尺寸生成合成河道，不分析真实地物。导出文件同样标记为模拟结果。</div></div>')

        gr.HTML('<footer class="app-footer"><span>WaterExtraction · 遥感水体提取工作台</span><span>V1.0 / 界面与基础交互版本</span></footer>')
        outputs = [state, preview, original, mask, overlay, info, stats, note, status, downloads, save_button]
        # 所有状态修改按同一队列顺序执行，防止清空后被旧任务重新填充。
        event_options = dict(concurrency_id="water_workspace", concurrency_limit=1, show_progress="minimal")
        gr.on([upload.upload, upload.clear], controller.upload, [upload], outputs, api_name="load_image", **event_options)
        parameters = [upload, model, threshold, alpha, color, boundary, state]
        run_button.click(controller.run, parameters, outputs, api_name="extract", **event_options)
        gr.on([model.input, threshold.release], controller.invalidate, [state], outputs, api_name="invalidate", **event_options)
        gr.on([alpha.release, color.change, boundary.change], controller.restyle, [upload, alpha, color, boundary, state], outputs, api_name="restyle", **event_options)
        save_button.click(controller.save, parameters, outputs, api_name="save_results", **event_options)
        clear_button.click(controller.clear, None, [upload, *outputs], api_name="clear", **event_options)
    return app.queue(default_concurrency_limit=1, max_size=16)
