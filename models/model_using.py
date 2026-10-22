"""U-Net++ 与 DeepLabv3+ 的自定义融合网络（单文件、纯 PyTorch）。

默认任务：RGB 图像的二分类语义分割，0=背景，1=前景（水体）。
默认示例：输入 [B, 3, 384, 384]，输出 [B, 1, 384, 384]。

结构设计：
1. 共享卷积编码器：提取原尺寸、1/2、1/4、1/8、1/16 五级特征。
2. ASPP：在 1/16 特征上提取多尺度上下文，供两个解码分支共同使用。
3. U-Net++ 分支：通过嵌套、密集跳跃连接恢复到原尺寸。
4. DeepLabv3+ 分支：将 ASPP 特征与 1/4 浅层特征融合、细化并上采样。
5. 融合两个分支的特征，用 1x1 卷积输出一个通道的 logits。

这是结合两种结构特点的自定义模型，不是任何一篇原论文的完整复现。
使用自定义卷积编码器，参数随机初始化，不加载预训练权重。
只返回最终预测张量，不返回辅助预测，不使用深监督。

训练示例：
    model = UNetPPDeepLabV3Plus()
    model.train()
    images = torch.randn(2, 3, 384, 384)
    targets = torch.randint(0, 2, (2, 1, 384, 384)).float()
    logits = model(images)
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    loss.backward()

注意：forward 返回未经过 Sigmoid 的 logits，可直接用于 BCEWithLogitsLoss。
推理时使用 torch.sigmoid(logits) 得到前景概率，再用 0.5 阈值得到 0/1 掩码。
标签应为浮点型 [B, 1, H, W]，取值为 0/1，而不是 0/255。

运行演示：python unetpp_deeplabv3plus.py
依赖：torch（请使用已经安装好的 CPU 或 CUDA 版 PyTorch）。

结构参考：
    UNet++: https://arxiv.org/abs/1807.10165
    DeepLabv3+: https://arxiv.org/abs/1802.02611
"""

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def resize_like(x: Tensor, reference: Tensor) -> Tensor:
    """按目标特征图尺寸上采样，避免奇数尺寸时出现拼接不匹配。"""
    return F.interpolate(
        x, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class ConvBNReLU(nn.Sequential):
    """卷积 + BatchNorm + ReLU，默认保持空间尺寸不变。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    """连续两个 3x3 卷积块，用于编码器和 U-Net++ 解码器。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )


class SeparableConv(nn.Sequential):
    """深度可分离卷积：逐通道 3x3 卷积，再用 1x1 卷积融合通道。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__(
            ConvBNReLU(in_channels, in_channels, dilation=dilation, groups=in_channels),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
        )


class ASPP(nn.Module):
    """空洞空间金字塔池化：1x1、三个空洞卷积分支、全局池化分支。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: Sequence[int] = (6, 12, 18),
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(in_channels, out_channels, kernel_size=1)]
            + [SeparableConv(in_channels, out_channels, rate) for rate in rates]
        )

        # 全局池化后为 [B, C, 1, 1]，此处分支不使用 BatchNorm，
        # 因此默认输入尺寸下，batch_size=1 也可以正常训练。
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBNReLU((len(rates) + 2) * out_channels, out_channels, kernel_size=1),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [branch(x) for branch in self.branches]
        features.append(resize_like(self.global_pool(x), x))
        return self.project(torch.cat(features, dim=1))


class NestedDecoder(nn.Module):
    """U-Net++ 嵌套解码器，x_i_j 表示第 i 个尺度、第 j 列的节点。

    每个节点融合：本尺度之前所有节点 + 下一级上一列节点的上采样结果。
    最深层输入为 ASPP 特征，最终输出 x0_4，空间尺寸与原图相同。
    """

    def __init__(self, channels: Sequence[int], context_channels: int):
        super().__init__()
        c0, c1, c2, c3 = channels[:4]

        # 第一列：j=1。
        self.conv0_1 = DoubleConv(c0 + c1, c0)
        self.conv1_1 = DoubleConv(c1 + c2, c1)
        self.conv2_1 = DoubleConv(c2 + c3, c2)
        self.conv3_1 = DoubleConv(c3 + context_channels, c3)

        # 第二列：j=2。
        self.conv0_2 = DoubleConv(c0 * 2 + c1, c0)
        self.conv1_2 = DoubleConv(c1 * 2 + c2, c1)
        self.conv2_2 = DoubleConv(c2 * 2 + c3, c2)

        # 第三列：j=3；第四列：j=4。
        self.conv0_3 = DoubleConv(c0 * 3 + c1, c0)
        self.conv1_3 = DoubleConv(c1 * 3 + c2, c1)
        self.conv0_4 = DoubleConv(c0 * 4 + c1, c0)

    def forward(self, features: Sequence[Tensor], context: Tensor) -> Tensor:
        x0_0, x1_0, x2_0, x3_0 = features

        x0_1 = self.conv0_1(torch.cat([x0_0, resize_like(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, resize_like(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, resize_like(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, resize_like(context, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, resize_like(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, resize_like(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, resize_like(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, resize_like(x1_2, x0_0)], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, resize_like(x2_2, x1_0)], dim=1)
        )
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, resize_like(x1_3, x0_0)], dim=1)
        )
        return x0_4


class DeepLabDecoder(nn.Module):
    """将 ASPP 特征上采样到 1/4 尺寸，与压缩后的浅层特征融合。"""

    def __init__(
        self,
        low_level_channels: int,
        context_channels: int,
        decoder_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.low_level_project = ConvBNReLU(low_level_channels, 48, kernel_size=1)
        self.refine = nn.Sequential(
            SeparableConv(context_channels + 48, decoder_channels),
            SeparableConv(decoder_channels, decoder_channels),
            # 先压缩通道，再上采样到原尺寸，减少融合阶段的显存占用。
            ConvBNReLU(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, context: Tensor, low_level: Tensor) -> Tensor:
        low_level = self.low_level_project(low_level)
        context = resize_like(context, low_level)
        return self.refine(torch.cat([context, low_level], dim=1))


class UNetPPDeepLabV3Plus(nn.Module):
    """共享编码器 + ASPP + 两个解码分支 + 特征融合分割头。

    默认参数下，输入 384x384 时的主要特征形状：
        编码器：32x384x384、64x192x192、128x96x96、256x48x48、512x24x24
        ASPP：256x24x24
        U-Net++ 分支：32x384x384
        DeepLabv3+ 分支：32x96x96，再上采样到 32x384x384
        特征融合：64x384x384 -> 32x384x384
        输出 logits：1x384x384

    base_channels 控制编码器及嵌套解码器的宽度，默认 32。
    输入尺寸不写死；默认演示使用 384x384，输出空间尺寸始终跟随输入。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        aspp_channels: int = 256,
        decoder_channels: int = 128,
        aspp_rates: Sequence[int] = (6, 12, 18),
    ):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        c0, c1, c2, c3, c4 = channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder0 = DoubleConv(in_channels, c0)
        self.encoder1 = DoubleConv(c0, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)

        self.aspp = ASPP(c4, aspp_channels, rates=aspp_rates)
        self.nested_decoder = NestedDecoder(channels, aspp_channels)
        self.deeplab_decoder = DeepLabDecoder(
            low_level_channels=c2,
            context_channels=aspp_channels,
            decoder_channels=decoder_channels,
            out_channels=c0,
        )
        self.fusion = DoubleConv(c0 * 2, c0)
        self.classifier = nn.Conv2d(c0, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.encoder0(x)  # [B, 32, 384, 384]
        x1 = self.encoder1(self.pool(x0))  # [B, 64, 192, 192]
        x2 = self.encoder2(self.pool(x1))  # [B, 128, 96, 96]
        x3 = self.encoder3(self.pool(x2))  # [B, 256, 48, 48]
        x4 = self.encoder4(self.pool(x3))  # [B, 512, 24, 24]

        context = self.aspp(x4)  # [B, 256, 24, 24]
        nested = self.nested_decoder((x0, x1, x2, x3), context)
        deeplab = self.deeplab_decoder(context, x2)
        deeplab = resize_like(deeplab, nested)

        fused = self.fusion(torch.cat([nested, deeplab], dim=1))
        return self.classifier(fused)  # [B, 1, 384, 384]，不加 Sigmoid


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetPPDeepLabV3Plus().to(device)
    model.eval()
    images = torch.randn(1, 3, 384, 384, device=device)

    with torch.inference_mode():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        masks = (probabilities >= 0.5).to(torch.uint8)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"设备：{device}")
    print(f"输入图像：{tuple(images.shape)}")
    print(f"输出 logits：{tuple(logits.shape)}")
    print(f"前景概率图：{tuple(probabilities.shape)}")
    print(f"二值掩码：{tuple(masks.shape)}，0=背景，1=前景")
    print(f"参数量：{parameter_count:,}（{parameter_count / 1_000_000:.2f} M）")
    print("这是随机初始化模型的形状演示；实际分割需要先训练或加载训练好的权重。")
