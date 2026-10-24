# WaterExtraction · 遥感影像水体智能提取系统

**V1.0 — Gradio 水体提取界面与独立的 CAID 训练模块。** 仓库名称为 `WaterExtractSoftware`，软件名称为 `WaterExtraction`；项目文件直接放在仓库根目录。

**训练与应用统一使用 `models/model_using.py` 中的 `UNetPPDeepLabV3Plus`。** 训练后，应用加载 `weights/best.pth` 预测真实影像；前端不提供算法选择，也不再生成模拟掩码。

## 安装与启动

使用 **Python 3.10 或更高版本**；已验证 Python 3.12、Gradio 6.30.0、PyTorch 2.7.1 / torchvision 0.22.1、SMP 0.5.0。训练默认使用 CUDA；请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与本机环境匹配的 PyTorch / torchvision，再安装项目依赖。应用可在 CPU 上运行，CUDA 可用时默认使用 GPU。

```bash
git clone https://github.com/LIKE9426334946/WaterExtractSoftware.git
cd WaterExtractSoftware
python -m venv .venv
```

Windows PowerShell（直接调用虚拟环境 Python，无需修改脚本执行策略）：

```powershell
.\.venv\Scripts\python.exe -m pip install gradio==6.30.0 segmentation-models-pytorch==0.5.0 numpy Pillow PyYAML tqdm
.\.venv\Scripts\python.exe app.py
```

Windows CMD 也可先执行 `.venv\Scripts\activate.bat`，再执行以下通用启动命令。Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install gradio==6.30.0 segmentation-models-pytorch==0.5.0 numpy Pillow PyYAML tqdm
python app.py
```

浏览器打开 **http://127.0.0.1:7860**。按 `Ctrl+C` 停止程序。默认仅监听本机，不创建公开分享链接。

首次使用先按下面的训练说明准备 CAID 并执行 `python train.py`。训练生成 `weights/best.pth` 后，执行 `python app.py`，登录后上传影像并点击“开始提取”。也可以通过 `inference.checkpoint` 指向已训练好的权重文件；`model` 和 `training.image_mean` / `image_std` 必须与训练时一致。没有权重时仍可登录和上传影像，点击提取会给出明确提示。

## 登录与注册

打开首页后显示登录页，背景直接使用 `ui/login.png`，右侧显示账号、密码及登录按钮，下方“立即注册”可切换到注册页。

- **默认账号**：`admin`，**默认密码**：`123456`。首次启动自动创建。
- **注册**：输入账号、密码和确认密码；账号为 3～32 位字母、数字、汉字或下划线，密码为 6～128 位。重复账号或两次密码不一致时会显示提示，注册成功后返回登录页。
- **登录**：成功后进入 `/app/` 选择首页，可选择“开始进行提取”或“查看保存结果”；刷新页面保留登录状态并回到选择首页。顶部“退出登录”返回登录页并使本次会话失效。
- **保存位置**：账号保存在 `data/users.json`，密码保存为随机盐与 PBKDF2-SHA256 哈希。注册账号重启后仍可使用；该目录已由 `.gitignore` 排除，不提交用户数据。
- **会话**：默认有效期 8 小时，服务重启后需要重新登录。通过 `auth.users_file` 和 `auth.session_hours` 配置存储位置与有效期；当前启动方式为单进程服务。

登录和注册页面复用 Gradio 已依赖的 FastAPI / Uvicorn，不增加额外第三方依赖。工作区使用 Gradio 的 `auth_dependency` 并配合服务端请求检查，未登录时禁止访问上传、推理、队列及下载接口。训练入口仍为独立的 `python train.py`。

## CAID 模型训练

### 数据集结构与标签

数据来源：[CAID 官方 Zenodo v2](https://zenodo.org/records/16461280)、[作者仓库](https://github.com/lubyant/CAID)。本仓库不包含数据集或预训练权重。官方压缩包采用以下结构，**`JPEGImages` 中的文件实际为 `.png`**：

```text
data/CAID/CAID/
├── JPEGImages/                  # 20,689 张 RGB PNG 影像
│   └── img_9132.png
├── SegmentationClass/           # 同名 PNG 标注
│   └── img_9132.png
└── ImageSets/Segmentation/
    ├── train.txt                # 14,482 个文件名，不含扩展名
    ├── val.txt                  # 3,103 个文件名，不含扩展名
    └── test.txt                 # 训练代码不读取
```

已检查官方 ZIP 文件目录、完整 train/val 清单及四组实际样本；train/val 清单没有交集。抽查的影像为 500×500 RGB，标注为单通道 0/1。按照官方定义，数据读取时明确将 **0 / 黑色映射为背景 0，非零水体像素映射为 1**；也支持作者给出的黑色背景、红色水体调色板标注。Mask 不做图像归一化，返回 `(1, H, W)` 的 `float32` Tensor。

下载并解压数据后，在 `config.yaml` 中设置 `dataset.root`，指向直接包含 `JPEGImages`、`SegmentationClass` 和 `ImageSets` 的目录。图像、标注通过清单中的同名 ID 配对，不按文件排列顺序配对。所有相对路径以项目根目录为基准，也可以设置绝对路径。

### 配置与运行

训练配置全部位于 `config.yaml` 的 `dataset`、`model`、`training` 三个部分。训练使用 `models/model_using.py` 中的 **UNetPPDeepLabV3Plus**：共享卷积编码器、ASPP、U-Net++ 与 DeepLabv3+ 双分支融合，随机初始化。默认采用 384×384 训练裁剪、batch size 8、100 epochs、AdamW 和 BCE 损失。在已安装依赖且数据就绪后运行：

```bash
python train.py
```

Windows 未激活虚拟环境时，可使用 `.\.venv\Scripts\python.exe train.py`。

- **设备**：默认 `training.device: cuda`、`amp: true`。CUDA 不可用时自动使用 CPU，也可设置 `device: cpu`；AMP 只在 CUDA 上启用。
- **模型**：`model` 配置直接传给 `UNetPPDeepLabV3Plus`，可调整 `base_channels`、`aspp_channels`、`decoder_channels` 和 `aspp_rates`。水体二分类保持 `in_channels: 3`、`out_channels: 1`，输出与输入同尺寸的原始 logits。模型工厂直接导入该网络，不再创建 SMP 网络。
- **初始化与归一化**：该网络使用自定义编码器，不加载 ImageNet 权重。RGB 先除以 255，再使用 `training.image_mean` / `image_std` 归一化；默认数值延续原训练配置的 ImageNet 均值和标准差，训练、验证使用相同设置。
- **损失**：`BCEWithLogitsLoss` 调用 SMP 的 `SoftBCEWithLogitsLoss(smooth_factor=None)`，等价于不平滑标签的 BCE；改为 `DiceLoss` 即调用 SMP 二分类 Dice 损失。

训练先按 `scale_range` 等比随机缩放，再按需填充、随机裁剪和水平/垂直翻转；图像和 Mask 共用几何变换，Mask 只使用最近邻插值。仅图像按配置归一化，返回 `(3, H, W)` Tensor。

验证直接输入完整影像，不随机增强、缩放或裁剪；网络内部按实际特征尺寸插值，输出与原始标注对齐，无需访问 SMP 编码器属性或额外填充。默认验证 batch size 为 1，可处理不同尺寸影像。每次验证汇总全部样本的 TP、FP、FN、TN，再调用 `smp.metrics` 的 `micro` 归约计算**水体前景** IoU、F1、Precision、Recall。预测按 `sigmoid(logits) > metric_threshold` 二值化，阈值默认 0.5；指标不是各 batch 指标的平均值。随机种子固定，验证预处理确定；不使用测试集选择模型。

### 训练输出

```text
weights/best.pth                 # 当前运行中验证 IoU 最佳的 model.state_dict()
outputs/train/history.csv        # 每个 epoch 写入并刷新一行
```

仅当验证 IoU **严格提高**时覆盖 `best.pth`，不保存每轮权重，也不保存优化器状态。终端显示 tqdm 进度、训练/验证损失、四项指标，并在结束时输出最佳 epoch 与 IoU。CSV 包含 `epoch`、`train_loss`、`val_loss`、`val_iou`、`val_f1`、`val_precision`、`val_recall`、`learning_rate`。

每次运行会重新创建训练日志，并在本次首次验证后写入最佳权重；若需保留多次实验，在运行前修改 `save_dir` 和 `output_dir`。数据集、生成的权重和运行日志均已加入 Git 忽略规则。

## Gradio 功能与操作

登录后先显示两个入口：“开始进行提取”和“查看保存结果”。标题位于左上角，顶部右侧显示软件版本和退出登录按钮。桌面布局随浏览器可用高度调整预览区域，三张结果图在同一屏并排展示。

1. **上传页**：从首页选择“开始进行提取”，再点击“上传图像”选择 JPG、JPEG、PNG，预览影像并显示文件名和像素尺寸。默认最大 20 MB / 1600 万像素，可在配置中修改。
2. **提取设置**：默认折叠，仅提供水体预测阈值（默认 0.5），不显示算法名称或模型下拉框。
3. **开始提取**：成功后切换到结果页，同时显示原始影像、黑白分割掩码和叠加结果。未上传、权重缺失或提取失败时留在上传页。网络输出经过 Sigmoid 和重叠区域概率平均后，按 `scores > threshold` 转为掩码，与训练验证使用相同的严格大于规则。
4. **结果页**：下方提供透明度、颜色、边界开关和五项统计。调整样式只重绘叠加图，不重复推理，不改变掩码和统计。开启边界时轮廓使用实色显示。
5. **保存结果**：将当前原图、掩码、叠加图及统计写入输出目录并加入历史列表，出现两个 PNG 和一个 ZIP 的下载链接。ZIP 内包含三张图、统计 JSON 和结果说明。更改显示设置后需要再次保存，每次保存产生独立记录。
6. **返回或清空**：“返回上传”保留当前图像，可更换影像、修改参数后再次提取；“清空结果”清除本会话图像、结果、统计和下载链接并返回上传页，保留当前参数和已保存的磁盘文件。“返回首页”回到两个功能入口，保留当前提取会话。
7. **查看保存结果**：首页进入历史列表，显示缩略图和从 **1** 开始的编号。点击编号或缩略图，显示该次保存的原始影像、分割掩码、叠加结果和统计，可再次下载；“返回列表”继续选择其他编号。“刷新列表”读取最新记录，数量较多时仅列表内部滚动。

更换影像或修改阈值会清除旧预测结果，必须重新提取。损坏文件、伪装扩展名、未上传影像、超限文件等会给出中文提示。各浏览器会话的未保存图像和当前提取结果互相独立；历史页展示当前 `files.output_dir` 中已显式保存的结果，所有登录账号共用该本地结果库。

编号存储在 `outputs/history.json`（随 `files.output_dir` 调整），刷新、退出登录或重启后保持不变，已移除记录的编号不会重新分配。升级前的 `prediction_*` 保存目录会自动加入列表；旧版未保存原图的记录使用叠加图作为缩略图，详情提示“旧记录未保存原始影像”，仍可查看已有掩码、叠加图和统计。新记录使用独立保存的 RGB 原图，上传缓存失效后仍可查看。历史查看不加载模型、不重新推理。

## 结果定义

| 项目 | 含义 |
| --- | --- |
| 掩码 | 单通道无损 PNG；0（黑色）为背景，255（白色）为水体 |
| 叠加图 | 原尺寸 RGB PNG，仅在预测水体区域混色；边界为可选的 4 邻域内轮廓 |
| 水体面积占比 | `水体像素数量 / 总像素数量 × 100%`，不代表真实地理面积 |
| 推理耗时 | 模型推理、输出校验与阈值化的耗时；首次提取包含权重加载，不含上传、图像显示及文件保存 |
| 像素分辨率 | 宽 × 高和总像素量（MP） |
| 地理分辨率 | 未提供地理参考，显示“未知”；不将 DPI 当作地面分辨率 |

图像不会被强制缩放成正方形。读取时应用 EXIF 方向，灰度转换为 RGB，透明像素合成到白底；掩码与叠加图对应处理后的完整像素尺寸。前端显示按容器适配，不改变导出分辨率。

保存目录示例：

```text
outputs/prediction_20261022_103000_<随机标识>/
├── prediction_original.png
├── prediction_mask.png
├── prediction_overlay.png
├── statistics.json
└── prediction_results.zip
```

文件名以 `prediction_` 开头，PNG 元数据和统计 JSON 标记为模型预测结果（`is_simulation: false`）。算法名称仅记录在导出的统计文件中。掩码本身不添加文字，以保证像素仅为 0/255。每次保存创建独立目录，避免覆盖。

## 文件职责

```text
WaterExtraction/                  # 克隆后的目录名默认是 WaterExtractSoftware
├── app.py                        # 配置、组装与启动
├── train.py                      # 独立训练、验证、日志与最佳权重保存
├── config.yaml                   # 界面及全部训练配置
├── data/users.json               # 首次启动生成的本地账号，不提交到仓库
├── models/
│   ├── __init__.py
│   ├── inference.py              # 权重加载、重叠分块推理与概率图
│   ├── model_factory.py          # 从配置创建 UNetPPDeepLabV3Plus
│   └── model_using.py            # 自定义 U-Net++ / DeepLabv3+ 融合网络
├── ui/
│   ├── __init__.py
│   ├── login.py                  # 登录、注册、退出及请求鉴权
│   ├── login.html                # 登录和注册页面模板
│   ├── login.css                 # 右侧表单与自适应布局
│   ├── login.png                 # 登录背景图
│   ├── layout.py                 # Gradio Blocks 布局与事件绑定
│   ├── callbacks.py              # 会话状态、交互与中文错误提示
│   ├── history.py                # 历史缩略图列表与只读结果详情
│   ├── presentation.py           # 标题、图像信息与统计卡片
│   └── styles.css
├── utils/
│   ├── __init__.py
│   ├── configuration.py          # 配置读取与校验
│   ├── authentication.py         # 账号哈希持久化与登录会话
│   ├── image_processing.py       # 图像读取、真实格式与尺寸校验
│   ├── dataset.py                # CAID 清单、标签映射及同步增强
│   ├── preprocessing.py          # 训练、验证及推理共用 RGB 归一化
│   ├── metrics.py                # SMP 统计量与全局指标
│   ├── pipeline.py               # 推理、阈值化、统计及重绘流程
│   ├── visualization.py          # 混色与边界计算
│   ├── history.py                # 本地结果扫描与持久编号
│   └── exporting.py              # PNG、JSON 与 ZIP 导出
├── tests/
│   ├── test_workflow.py          # 核心业务和状态回归检查
│   ├── test_gradio_api.py        # 真实 HTTP / Gradio 会话与下载流程
│   ├── test_training.py          # 数据语义、全局指标与最佳权重规则
│   ├── test_inference.py         # 权重加载、归一化及分块覆盖
│   ├── test_history.py           # 保存快照、编号持久化与旧记录读取
│   └── test_authentication.py    # 默认账号、注册、会话和接口鉴权
├── weights/                      # 最佳模型权重，不提交生成文件
└── outputs/
    ├── history.json              # 运行时生成的保存记录编号索引
    └── train/                    # 训练 CSV；其他子目录保存应用预测结果
```

## 应用推理配置

`inference` 配置示例：

```yaml
inference:
  checkpoint: weights/best.pth
  device: cuda
  amp: true
  tile_size: 384
  tile_overlap: 64
  threshold: 0.5
```

- `checkpoint` 是 `train.py` 保存的纯 `state_dict` 文件，相对路径以配置文件所在目录为基准。修改训练的 `save_dir` 后，相应更新此路径。
- 首次点击提取时以 `weights_only=True` 加载并严格校验权重，随后缓存模型并使用 `eval()`、`torch.inference_mode()`；重新训练或替换权重后，重启应用加载新权重。缺失或结构不匹配的权重会阻止提取。
- `device` 默认 CUDA，不可用时使用 CPU。`amp` 只在 CUDA 上启用。
- 按 `tile_size` 分块逐块预测，重叠宽度由 `tile_overlap` 指定；重叠区域取概率平均，最后统一阈值化。小图用边缘复制填充，输出裁回原尺寸；不缩放或拉伸原图。分块推理的上下文与训练验证阶段的整图前向不同，结果可能存在差异。
- 训练、验证和推理复用 `utils/preprocessing.py`，按相同 RGB 顺序、均值与标准差处理。后处理、统计和导出位于 `utils/pipeline.py` 和 `utils/exporting.py`，与网络代码分离。

## 验证

在已安装依赖的环境中，从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

覆盖：二值像素与原尺寸、阈值端点、叠加透明度、背景保持、边界、重绘后统计不变、PNG/ZIP 导出、预测元数据、格式/尺寸校验、灰度与透明通道、EXIF 旋转、清空/换图/参数变更状态、过期结果禁止保存，以及推理接口替换。

另通过临时启动的本机 Gradio 服务验证上传、真实权重加载、提取、重绘、文件下载、阈值端点、换图、损坏上传、清空和会话隔离。测试结束自动关闭服务，使用临时输出目录；HTTP 检查使用临时的小通道网络权重，仅验证功能流程。

应用推理检查还覆盖权重只加载一次、权重缺失或不匹配时拒绝提取、与直接网络前向结果一致、非整块尺寸的重叠合并及小图填充。曾使用下述训练检查生成的默认网络权重完成一张 500×500 CAID 影像的应用预测，验证掩码、叠加图、统计和 PNG/ZIP 导出。

加入登录后，25 项自动检查通过，覆盖默认账号、错误密码、注册校验、账号持久化、密码哈希、会话过期及退出失效、未登录接口拦截和登录后的完整提取下载流程。另通过 Chromium 验证登录/注册切换、新账号登录、刷新保持登录、提取保存和退出；1920×1080、1366×768、1024×600 下登录页、注册页和保存后工作区均无横向或纵向页面滚动。

训练回归检查覆盖标签映射、图像/Mask 同步裁剪翻转、归一化、确定性验证、全局指标汇总、原尺寸验证、两种损失反向传播、IoU 持平或下降时不保存，以及 CSV 字段。融合网络另检查奇数输入尺寸及两个解码分支的梯度。

已使用默认通道配置的 `UNetPPDeepLabV3Plus`，在官方训练集两张、验证集两张实际样本上完成一轮 CPU 训练（batch size 1）：384×384 裁剪、完整 500×500 验证、参数更新、`best.pth` 严格重新加载和 CSV 写入均通过。未进行完整 CAID 训练或 CUDA AMP 实机验证，流程检查不代表模型分割精度。

增加选择首页和历史查看后，31 项自动检查通过，新增检查覆盖原图快照、ZIP 内容、编号持久化、并发保存、删除后不复用编号、旧记录读取、空列表和保存失败处理。HTTP 检查同时覆盖首页初始显示、跨会话历史读取以及清空后历史仍保留。

已通过 Chromium 验证两个首页入口、编号点击、三图与保存文件一致、历史下载、重复选择、清空/刷新/退出重登/服务重启后的历史保留，以及 20 条记录的列表浏览。1920×1080、1366×768、1024×600 下，首页、上传设置页、保存后提取页、历史列表和详情页均无需整体页面滚动；大量历史记录仅在列表内部滚动。

浏览器验收建议：登录 → 开始进行提取 → 未上传直接提取（应留在上传页）→ 上传 JPG/PNG → 开始提取（进入三图结果页）→ 修改透明度/颜色/边界 → 保存并下载 → 返回首页 → 查看保存结果 → 点击编号 → 返回列表 → 刷新或重新登录后查看。检查调整窗口大小、展开设置及显示下载链接后仍能看全页面，并确认显示与导出对应同一次保存。

两页布局此前已在 Chromium 中验证上传、页面切换、返回、换图、样式更新、保存下载和清空。窗口检查覆盖 1920×1080、1366×768、1280×720、1280×600、1024×600；首页、展开设置及保存后结果页均无需页面纵向滚动。

接入真实权重后，已再次通过 Chromium 检查上传、提取、三图展示、保存和清空；1366×768、1024×600 下的展开设置页和保存后结果页均无横向或纵向页面滚动，页面不包含模型选择或算法名称。

## 常见问题

- **7860 端口被占用**：停止占用进程，或在 `config.yaml` 的 `server.port` 指定其他端口。
- **本机代理导致无法打开/启动**：为 `127.0.0.1,localhost` 配置代理排除项。若代理使用 SOCKS 且提示缺少 `socksio`，按提示执行 `python -m pip install "httpx[socks]"`；普通环境无需额外安装。
- **无法保存**：检查 `files.output_dir` 指向的目录是否可写，以及磁盘空间是否充足。相对路径以配置文件所在目录为基准。

当前通过 `gr.mount_gradio_app()` 传入 `theme`、`css_paths` 等参数，将工作区挂载到带登录鉴权的 FastAPI 应用。参考：[Gradio 挂载应用](https://www.gradio.app/docs/gradio/mount_gradio_app)、[Gradio 自定义样式](https://www.gradio.app/guides/custom-CSS-and-JS)。
