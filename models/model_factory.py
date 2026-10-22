"""创建 model_using.py 中的融合网络，输出单通道原始 logits。"""

from .model_using import UNetPPDeepLabV3Plus


def create_model(config):
    return UNetPPDeepLabV3Plus(**config)
