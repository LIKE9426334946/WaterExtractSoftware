"""权重加载、共用归一化及分块合并检查，不下载权重。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models import create_predictor
from models.model_factory import create_model
from utils.configuration import load_config
from utils.preprocessing import image_to_tensor

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = load_config(ROOT / "config.yaml")
        self.config = replace(
            base, checkpoint=Path(directory.name) / "best.pth", device="cpu",
            tile_size=32, tile_overlap=8,
            model=dict(base.model, base_channels=4, aspp_channels=8, decoder_channels=8),
        )

    def test_checkpoint_matches_direct_network_and_is_loaded_once(self):
        torch.manual_seed(42)
        model = create_model(self.config.model).eval()
        torch.save(model.state_dict(), self.config.checkpoint)
        image = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        tensor = image_to_tensor(image, self.config.image_mean, self.config.image_std)
        with torch.inference_mode():
            expected = torch.sigmoid(model(tensor.unsqueeze(0)))[0, 0].numpy()
        predictor = create_predictor(self.config)
        with patch("models.inference.torch.load", wraps=torch.load) as load:
            first = predictor.predict(image)
            second = predictor.predict(image)
        self.assertEqual(load.call_count, 1)
        self.assertFalse(predictor.model.training)
        self.assertFalse(predictor.amp)
        self.assertEqual(first.model_name, "UNetPPDeepLabV3Plus")
        np.testing.assert_allclose(first.scores, expected, atol=1e-6)
        np.testing.assert_array_equal(first.scores, second.scores)

    def test_overlap_and_small_image_padding_preserve_every_pixel(self):
        class PixelModel(torch.nn.Module):
            def forward(self, image):
                if not torch.is_inference_mode_enabled():
                    raise RuntimeError("inference_mode 未启用")
                return image[:, :1]

        torch.save(PixelModel().state_dict(), self.config.checkpoint)
        with patch("models.inference.create_model", return_value=PixelModel()):
            predictor = create_predictor(self.config)
            for height, width in ((37, 53), (7, 11), (32, 65)):
                image = np.random.default_rng(12).integers(0, 256, (height, width, 3), dtype=np.uint8)
                probability = predictor.predict(image).scores
                normalized = (image[..., 0] / 255.0 - self.config.image_mean[0]) / self.config.image_std[0]
                expected = 1 / (1 + np.exp(-normalized))
                self.assertEqual(probability.shape, (height, width))
                self.assertEqual(probability.dtype, np.float32)
                np.testing.assert_allclose(probability, expected, atol=1e-6)

    def test_missing_and_incompatible_weights_do_not_create_predictions(self):
        predictor = create_predictor(self.config)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "未找到模型权重"):
            predictor.predict(image)
        self.assertIsNone(predictor.model)
        torch.save({"wrong_layer": torch.zeros(1)}, self.config.checkpoint)
        with self.assertRaisesRegex(ValueError, "权重与模型结构不匹配"):
            predictor.predict(image)
        self.assertIsNone(predictor.model)


if __name__ == "__main__":
    unittest.main()
