"""训练数据语义、全局指标与权重保存规则的回归检查，无需下载模型权重。"""

import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

import train
from models.model_using import UNetPPDeepLabV3Plus
from utils.dataset import CAIDDataset
from utils.metrics import batch_statistics, compute_metrics


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = yaml.safe_load(train.CONFIG_PATH.read_text(encoding="utf-8"))
        self.config["dataset"]["root"] = str(self.root)
        self.config["training"].update(
            image_size=16, scale_range=[1.0, 1.0],
            horizontal_flip_prob=1.0, vertical_flip_prob=1.0,
            device="cpu", epochs=4, batch_size=1,
            save_dir=str(self.root / "weights"), output_dir=str(self.root / "logs"),
        )
        for folder in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
            (self.root / folder).mkdir(parents=True)
        self.water = np.zeros((13, 21), dtype=np.uint8)
        self.water[2:11, 5:17] = 1
        for split in ("train", "val"):
            (self.root / f"ImageSets/Segmentation/{split}.txt").write_text(split + "\n")
            image = np.repeat((self.water * 255)[..., None], 3, axis=2)
            Image.fromarray(image).save(self.root / f"JPEGImages/{split}.png")
            if split == "train":
                mask = Image.fromarray(self.water * 255)
            else:
                mask = Image.fromarray(self.water).convert("P")
                mask.putpalette([0, 0, 0, 128, 0, 0] + [0] * 762)
            mask.save(self.root / f"SegmentationClass/{split}.png")

    def test_mapping_paired_geometry_and_deterministic_validation(self):
        training = CAIDDataset(self.config, "train")
        validation = CAIDDataset(self.config, "val")
        self.assertEqual(training.ids, ["train"])
        self.assertEqual(validation.ids, ["val"])
        random.seed(17)
        image, mask = training[0]
        self.assertEqual(image.shape, (3, 16, 16))
        self.assertEqual(mask.shape, (1, 16, 16))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        self.assertTrue(torch.equal(image[0] > 0, mask[0] > 0))
        first = validation[0]
        random.seed(99)
        second = validation[0]
        self.assertEqual(first[0].shape, (3, 13, 21))
        self.assertTrue(torch.equal(first[0], second[0]))
        np.testing.assert_array_equal(first[1][0].numpy(), self.water)
        self.assertEqual(first[0].dtype, torch.float32)
        mean = torch.tensor(self.config["training"]["image_mean"])[:, None, None]
        std = torch.tensor(self.config["training"]["image_std"])[:, None, None]
        expected = torch.tensor(self.water).expand(3, -1, -1).float()
        torch.testing.assert_close(first[0], (expected - mean) / std)

    def test_custom_network_preserves_odd_size_and_backpropagates_both_branches(self):
        config = dict(self.config["model"], base_channels=4, aspp_channels=8, decoder_channels=8)
        model = train.create_model(config)
        self.assertIsInstance(model, UNetPPDeepLabV3Plus)
        image = torch.randn(1, 3, 33, 47)
        mask = torch.randint(0, 2, (1, 1, 33, 47)).float()
        logits = model(image)
        self.assertEqual(logits.shape, mask.shape)
        loss = train.create_loss("BCEWithLogitsLoss")(logits, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for block in (model.encoder0, model.aspp, model.nested_decoder, model.deeplab_decoder, model.classifier):
            gradients = [parameter.grad for parameter in block.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))

    def test_metrics_use_global_counts_and_strict_threshold(self):
        logits = torch.tensor([[[[10., 0., -10., -10.]]], [[[10., 10., 10., 10.]]]])
        masks = torch.tensor([[[[1., 0., 0., 0.]]], [[[1., 0., 0., 0.]]]])
        totals = sum(batch_statistics(logits[i:i + 1], masks[i:i + 1]) for i in range(2))
        self.assertEqual(totals.tolist(), [2, 3, 0, 3])
        self.assertTrue(torch.equal(totals, batch_statistics(logits, masks)))
        metrics = compute_metrics(totals)
        self.assertAlmostEqual(metrics["iou"], 0.4)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        self.assertAlmostEqual(metrics["precision"], 0.4)
        self.assertEqual(metrics["recall"], 1.0)

    def test_validation_keeps_original_size_and_losses_support_backward(self):
        class PixelModel(torch.nn.Module):
            def forward(self, image):
                self.input_shape = image.shape[-2:]
                return image[:, :1]

        model = PixelModel()
        mask = torch.tensor(self.water, dtype=torch.float32)[None, None]
        image = (mask * 20 - 10).repeat(1, 3, 1, 1)
        criterion = train.create_loss("BCEWithLogitsLoss")
        loss, metrics = train.validate(model, [(image, mask)], criterion, torch.device("cpu"), False, 0.5, 1)
        self.assertEqual(model.input_shape, (13, 21))
        self.assertFalse(model.training)
        self.assertEqual(metrics, dict(iou=1.0, f1=1.0, precision=1.0, recall=1.0))
        self.assertAlmostEqual(loss, torch.nn.functional.binary_cross_entropy_with_logits(image[:, :1], mask).item())
        for name in ("BCEWithLogitsLoss", "DiceLoss"):
            logits = image[:, :1].clone().requires_grad_()
            train.create_loss(name)(logits, mask).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_only_strict_iou_improvement_saves_state_dict_and_csv(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        model = torch.nn.Conv2d(3, 1, 1)
        metrics = [dict(iou=value, f1=value, precision=value, recall=value) for value in (0.4, 0.4, 0.3, 0.5)]
        with patch.object(train, "CONFIG_PATH", config_path), \
                patch.object(train, "create_model", return_value=model), \
                patch.object(train, "train_one_epoch", return_value=0.7), \
                patch.object(train, "validate", side_effect=[(0.6, item) for item in metrics]), \
                patch.object(train.torch, "save") as save:
            train.main()
        self.assertEqual(save.call_count, 2)
        self.assertEqual(set(save.call_args.args[0]), set(model.state_dict()))
        self.assertEqual(save.call_args.args[1], self.root / "weights/best.pth")
        with (self.root / "logs/history.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["val_iou"]) for row in rows], [0.4, 0.4, 0.3, 0.5])
        self.assertEqual(len(rows[0]), 8)


if __name__ == "__main__":
    unittest.main()
