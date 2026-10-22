# WaterExtraction · 遥感影像水体智能提取系统

**V1.0 — Gradio 界面与基础交互版本。** 仓库名称为 `WaterExtractSoftware`，软件名称为 `WaterExtraction`；项目文件直接放在仓库根目录。

> **当前没有训练、加载或运行任何深度学习模型。所有掩码、叠加图和统计都是模拟演示结果，不能作为真实水体提取结果或模型性能依据。** 四个模型选项共用同一个模拟后端。

## 安装与启动

使用 **Python 3.10 或更高版本**；本项目在 Python 3.12、Gradio 6.30.0 上验证。无需 GPU、CUDA、PyTorch、OpenCV、数据库或 Node.js。

```bash
git clone https://github.com/LIKE9426334946/WaterExtractSoftware.git
cd WaterExtractSoftware
python -m venv .venv
```

Windows PowerShell（直接调用虚拟环境 Python，无需修改脚本执行策略）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Windows CMD 也可先执行 `.venv\Scripts\activate.bat`，再执行以下通用启动命令。Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 **http://127.0.0.1:7860**。按 `Ctrl+C` 停止程序。默认仅监听本机，不创建公开分享链接。

## 功能与操作

1. **上传影像**：选择 JPG、JPEG、PNG；显示缩略预览、原始影像、宽高及像素分辨率。默认最大 20 MB / 1600 万像素，可在配置中修改。
2. **选择模型与阈值**：U-Net、U-Net++、DeepLabv3+、SegFormer；阈值默认 0.5，范围 0～1。模型选项暂不影响模拟掩码。
3. **开始提取**：根据图像宽高生成确定性的合成河道分数图，按照 `scores >= threshold` 转换为黑白掩码。分数与真实影像内容无关，也不是模型预测概率。默认模拟分数落在 0 与 1 之间，阈值为 0 时全白、1 时全黑。
4. **调整显示**：透明度、颜色和边界显示会重绘叠加图，不重复推理，不改变掩码、像素统计及原提取耗时。透明度为 0 时完全透明，为 1 时完全着色；开启边界时轮廓始终用实色显示。
5. **保存结果**：将当前结果写入配置的输出目录，同时出现两个 PNG 和一个 ZIP 的下载链接。ZIP 内含掩码、叠加图、统计 JSON 和模拟说明。更改显示设置后需要再次保存，旧下载链接会隐藏。
6. **清空结果**：清除上传文件、所有图像、统计、下载链接与本次会话状态，保留当前参数和已保存的磁盘文件。

更换影像、修改模型或阈值会清除旧预测结果，必须重新提取。损坏文件、伪装扩展名、未上传影像、超限文件等会给出中文提示。不同浏览器会话不共用图像或结果。

## 结果定义

| 项目 | 含义 |
| --- | --- |
| 掩码 | 单通道无损 PNG；0（黑色）为背景，255（白色）为水体 |
| 叠加图 | 原尺寸 RGB PNG，仅在模拟水体区域混色；边界为可选的 4 邻域内轮廓 |
| 水体面积占比 | `水体像素数量 / 总像素数量 × 100%`，不代表真实地理面积 |
| 推理耗时 | 实际测量模拟后端、输出校验与阈值化的耗时；不含上传、图像显示及文件保存 |
| 像素分辨率 | 宽 × 高和总像素量（MP） |
| 地理分辨率 | 未提供地理参考，显示“未知”；不将 DPI 当作地面分辨率 |

图像不会被强制缩放成正方形。读取时应用 EXIF 方向，灰度转换为 RGB，透明像素合成到白底；掩码与叠加图对应处理后的完整像素尺寸。前端显示按容器适配，不改变导出分辨率。

保存目录示例：

```text
outputs/simulated_20261022_103000_<随机标识>/
├── simulated_mask.png
├── simulated_overlay.png
├── statistics.json
└── simulated_results.zip
```

文件名、PNG 文本元数据、JSON 的 `is_simulation` 和 ZIP 内说明均标记模拟性质。掩码本身不添加文字，以保证像素仅为 0/255。每次保存创建独立目录，避免覆盖。

## 文件职责

```text
WaterExtraction/                  # 克隆后的目录名默认是 WaterExtractSoftware
├── app.py                        # 配置、组装与启动
├── config.yaml                   # 服务、阈值、显示及文件配置
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── inference.py              # Predictor 接口及模拟实现
├── ui/
│   ├── __init__.py
│   ├── layout.py                 # Gradio Blocks 布局与事件绑定
│   ├── callbacks.py              # 会话状态、交互与中文错误提示
│   ├── presentation.py           # 标题、图像信息与统计卡片
│   └── styles.css
├── utils/
│   ├── __init__.py
│   ├── configuration.py          # 配置读取与校验
│   ├── image_processing.py       # 图像读取、真实格式与尺寸校验
│   ├── pipeline.py               # 推理、阈值化、统计及重绘流程
│   ├── visualization.py          # 混色与边界计算
│   └── exporting.py              # PNG、JSON 与 ZIP 导出
├── tests/
│   ├── test_workflow.py          # 核心业务和状态回归检查
│   └── test_gradio_api.py        # 真实 HTTP / Gradio 会话与下载流程
└── outputs/                      # 保存结果，不提交运行生成的文件
```

## 后续接入真实模型

实现 `models/inference.py` 中的 `Predictor` 协议，并在 `create_predictor()` 工厂中注册真实后端：

```python
def predict(self, image: np.ndarray, model_name: str) -> Prediction:
    # 输入：H×W×3 uint8 RGB。
    # 在此根据 model_name 选择模型，完成预处理、torch.inference_mode()、
    # 前向传播、sigmoid/softmax 以及分数图恢复到原图尺寸。
    return Prediction(
        scores=water_probability,  # H×W float32，有限数值，范围 [0, 1]
        backend_name="PyTorch / CAID",
        is_simulation=False,
    )
```

业务层统一阈值化、统计和导出，不与 Gradio 耦合。真实实现应依据训练配置处理颜色通道、归一化、填充/滑窗及尺寸恢复，并在模型端执行 `eval()`。目前工厂只接受 `simulation`，不会将未实现的真实后端静默替换成模拟结果。接入真实后端时同步将 `ui/layout.py` 的静态模拟横幅、图像标签及引导文案改为对应模式；页面交互和显示逻辑可以复用。

## 验证

在已安装依赖的环境中，从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

覆盖：二值像素与原尺寸、阈值端点、叠加透明度、背景保持、边界、重绘后统计不变、PNG/ZIP 导出、模拟元数据、格式/尺寸校验、灰度与透明通道、EXIF 旋转、清空/换图/参数变更状态、过期结果禁止保存，以及推理接口替换。

另通过临时启动的本机 Gradio 服务验证上传、四个模型选项、提取、重绘、文件下载、阈值端点、换图、损坏上传、清空和会话隔离。测试结束自动关闭服务，使用临时输出目录；这些检查不替代浏览器外观验收。

浏览器验收建议：未上传直接提取 → 上传任意 JPG/PNG → 开始提取 → 修改透明度/颜色/边界 → 保存并下载 → 修改阈值并重新提取 → 换图 → 清空。确认页面与导出始终对应同一次操作。

## 常见问题

- **7860 端口被占用**：停止占用进程，或在 `config.yaml` 的 `server.port` 指定其他端口。
- **本机代理导致无法打开/启动**：为 `127.0.0.1,localhost` 配置代理排除项。若代理使用 SOCKS 且提示缺少 `socksio`，按提示执行 `python -m pip install "httpx[socks]"`；普通环境无需额外安装。
- **无法保存**：检查 `files.output_dir` 指向的目录是否可写，以及磁盘空间是否充足。相对路径以配置文件所在目录为基准。

Gradio 6 的 `theme`、`css_paths` 等参数放在 `launch()` 中。参考：[Gradio 自定义样式](https://www.gradio.app/guides/custom-CSS-and-JS)、[Gradio 发行页](https://pypi.org/project/gradio/6.30.0/)。
