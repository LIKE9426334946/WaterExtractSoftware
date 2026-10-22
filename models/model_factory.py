"""直接调用 SMP 创建分割模型；输出原始 logits。"""

import segmentation_models_pytorch as smp


def create_model(config):
    return smp.create_model(
        arch=config["name"],
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=None,
    )
