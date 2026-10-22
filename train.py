"""水体二分类训练入口：直接运行 python train.py，配置读取自 config.yaml。"""

import csv
import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_factory import create_model
from utils.dataset import CAIDDataset
from utils.metrics import batch_statistics, compute_metrics

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def create_loss(name):
    return {
        "BCEWithLogitsLoss": smp.losses.SoftBCEWithLogitsLoss(smooth_factor=None),
        "DiceLoss": smp.losses.DiceLoss(mode="binary", from_logits=True),
    }[name]


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, epoch):
    model.train()
    loss_sum, pixel_count = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} 训练", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        loss_sum += loss_value * masks.numel()
        pixel_count += masks.numel()
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return loss_sum / pixel_count


@torch.inference_mode()
def validate(model, loader, criterion, device, amp, threshold, epoch):
    model.eval()
    loss_sum, pixel_count = 0.0, 0
    totals = torch.zeros(4, dtype=torch.int64, device=device)
    stride = model.encoder.output_stride
    for images, masks in tqdm(loader, desc=f"Epoch {epoch} 验证", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = masks.shape[-2:]
        # 完整影像验证：只填充输入，前向后裁回原尺寸，填充像素不参与损失和指标。
        images = F.pad(images, (0, -width % stride, 0, -height % stride), mode="replicate")
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)[..., :height, :width]
            loss = criterion(logits, masks)

        loss_sum += loss.item() * masks.numel()
        pixel_count += masks.numel()
        totals += batch_statistics(logits, masks, threshold)
    return loss_sum / pixel_count, compute_metrics(totals)


def main():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    training = config["training"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(training["device"])
    amp = training["amp"] and device.type == "cuda"
    train_dataset = CAIDDataset(config, "train")
    val_dataset = CAIDDataset(config, "val")
    train_loader = DataLoader(
        train_dataset, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training["val_batch_size"], shuffle=False,
        num_workers=training["num_workers"], pin_memory=device.type == "cuda",
    )
    model = create_model(config["model"]).to(device)
    criterion = create_loss(training["loss"])
    optimizer = getattr(torch.optim, training["optimizer"])(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    save_dir = ROOT / training["save_dir"]
    output_dir = ROOT / training["output_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"训练集：{len(train_dataset)} 张 | 验证集：{len(val_dataset)} 张 | 设备：{device} | CUDA AMP：{amp}")

    fields = ["epoch", "train_loss", "val_loss", "val_iou", "val_f1", "val_precision", "val_recall", "learning_rate"]
    best_iou, best_epoch = -1.0, 0
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, training["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, epoch)
            val_loss, metrics = validate(model, val_loader, criterion, device, amp, training["metric_threshold"], epoch)
            if metrics["iou"] > best_iou:
                best_iou, best_epoch = metrics["iou"], epoch
                torch.save(model.state_dict(), save_dir / "best.pth")
                print(f"已更新 best.pth：Epoch {epoch}，val_iou={best_iou:.6f}")

            learning_rate = optimizer.param_groups[0]["lr"]
            writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                **{f"val_{name}": value for name, value in metrics.items()},
                "learning_rate": learning_rate,
            })
            stream.flush()
            print(
                f"Epoch {epoch}/{training['epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"IoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f} | "
                f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | lr={learning_rate:g}"
            )
    print(f"训练结束：最佳 Epoch={best_epoch}，验证集水体 IoU={best_iou:.6f}")


if __name__ == "__main__":
    main()
