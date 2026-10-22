"""核心回归检查：python -m unittest discover -s tests -v。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from models import Prediction
from ui.callbacks import Controller, SessionData
from utils.configuration import load_config
from utils.exporting import save_result
from utils.image_processing import load_image
from utils.pipeline import extract, redraw
from utils.visualization import inner_boundary

ROOT = Path(__file__).resolve().parents[1]


class GradientPredictor:
    """测试业务流程用的确定性概率图，不用于应用。"""

    def predict(self, image):
        scores = np.linspace(0.05, 0.95, image.shape[1], dtype=np.float32)
        return Prediction(np.broadcast_to(scores, image.shape[:2]).copy(), "test predictor")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "input.png"
        self.rgb = np.full((80, 120, 3), [100, 140, 80], dtype=np.uint8)
        Image.fromarray(self.rgb).save(self.path)
        self.source = load_image(self.path, 20, 1000000)
        self.predictor = GradientPredictor()
        self.config = replace(load_config(ROOT / "config.yaml"), output_dir=self.root / "outputs")

    def result(self, threshold=0.5, alpha=0.5, color="#2563EB", boundary=False):
        return extract(self.source, self.predictor, threshold, alpha, color, boundary)

    def test_masks_and_statistics(self):
        result = self.result()
        self.assertEqual(result.mask.shape, (80, 120))
        self.assertEqual(set(np.unique(result.mask)), {0, 255})
        self.assertEqual(result.water_pixels + result.background_pixels, 80 * 120)
        self.assertAlmostEqual(result.water_percent, (result.mask == 255).mean() * 100)
        self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertEqual(result.model_name, "test predictor")

    def test_threshold_extremes_and_monotonicity(self):
        self.assertEqual(self.result(0).water_pixels, 9600)
        self.assertEqual(self.result(1).water_pixels, 0)
        self.assertGreater(self.result(0.2).water_pixels, self.result(0.8).water_pixels)
        for value in [-0.01, 1.01, float("nan")]:
            with self.assertRaises(ValueError):
                self.result(value)

    def test_overlay_alpha_endpoints_and_unchanged_background(self):
        transparent = self.result(alpha=0)
        np.testing.assert_array_equal(transparent.overlay, self.rgb)
        opaque = self.result(alpha=1, color="#FF0000")
        np.testing.assert_array_equal(opaque.overlay[opaque.mask == 0], self.rgb[opaque.mask == 0])
        self.assertTrue(np.all(opaque.overlay[opaque.mask != 0] == [255, 0, 0]))
        self.assertFalse(np.shares_memory(opaque.overlay, self.rgb))

    def test_restyle_keeps_mask_counts_and_timing(self):
        result = self.result()
        restyled = redraw(result, 0.8, "#00FF00", True)
        np.testing.assert_array_equal(restyled.mask, result.mask)
        self.assertEqual(restyled.water_pixels, result.water_pixels)
        self.assertEqual(restyled.elapsed_ms, result.elapsed_ms)
        self.assertTrue(np.all(restyled.overlay[inner_boundary(result.mask)] == [0, 255, 0]))

    def test_export_png_values_metadata_zip_and_unique_paths(self):
        result = self.result()
        paths = save_result(result, self.config.output_dir)
        with Image.open(paths[0]) as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (120, 80))
            self.assertEqual(mask.info["Simulation"], "false")
            np.testing.assert_array_equal(np.asarray(mask), result.mask)
        with Image.open(paths[1]) as overlay:
            np.testing.assert_array_equal(np.asarray(overlay), result.overlay)
        with ZipFile(paths[2]) as archive:
            metadata = json.loads(archive.read("statistics.json"))
            self.assertFalse(metadata["is_simulation"])
            self.assertEqual(metadata["model"], "test predictor")
            self.assertIsNone(metadata["spatial_resolution"])
            self.assertEqual(metadata["water_pixels"], result.water_pixels)
            self.assertIn("说明.txt", archive.namelist())
        self.assertNotEqual(paths, save_result(result, self.config.output_dir))

    def test_bad_uploads_and_limits(self):
        for filename, content in [("bad.txt", b"text"), ("corrupt.png", b"not png")]:
            path = self.root / filename
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                load_image(path, 20, 1000000)
        spoofed = self.root / "spoofed.png"
        Image.new("RGB", (10, 10)).save(spoofed, format="GIF")
        with self.assertRaises(ValueError):
            load_image(spoofed, 20, 1000000)
        with self.assertRaises(ValueError):
            load_image(self.path, 20, 100)
        with self.assertRaises(ValueError):
            load_image(self.path, 0, 1000000)

    def test_rgb_grayscale_transparent_and_jpeg_exif(self):
        gray = self.root / "gray.PNG"
        Image.new("L", (20, 30), 71).save(gray)
        self.assertEqual(load_image(gray, 20, 1000000).rgb.shape, (30, 20, 3))
        transparent = self.root / "alpha.png"
        Image.new("RGBA", (20, 30), (0, 0, 0, 0)).save(transparent)
        self.assertTrue(np.all(load_image(transparent, 20, 1000000).rgb == 255))
        jpeg = self.root / "rotated.JPEG"
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(self.rgb).save(jpeg, exif=exif)
        self.assertEqual(load_image(jpeg, 20, 1000000).rgb.shape, (120, 80, 3))

    def test_clear_replacement_and_corrupt_upload_clear_result(self):
        controller = Controller(self.config, self.predictor)
        populated = SessionData(self.source, self.result())
        cleared = controller.clear()
        self.assertIsNone(cleared[0])
        self.assertEqual(cleared[1], SessionData())
        self.assertIsNone(cleared[4])
        replacement = controller.upload(str(self.path))
        self.assertIsNone(replacement[0].result)
        self.assertIsNone(replacement[3])
        invalidated = controller.invalidate(populated)
        self.assertIsNone(invalidated[0].result)
        bad = self.root / "corrupt.png"
        bad.write_bytes(b"bad")
        with patch("ui.callbacks.gr.Warning"):
            invalid_upload = controller.upload(str(bad))
        self.assertEqual(invalid_upload[0], SessionData())
        self.assertFalse(invalid_upload[-1]["interactive"])

    def test_empty_and_stale_save_cannot_export(self):
        controller = Controller(self.config, self.predictor)
        with patch("ui.callbacks.gr.Warning"):
            empty = controller.run(None, 0.5, 0.5, "#2563EB", False, SessionData())
            self.assertIsNone(empty[0].result)
            controller.save(str(self.path), 0.9, 0.5, "#2563EB", False, SessionData(self.source, self.result()))
            controller.save("another.png", 0.5, 0.5, "#2563EB", False, SessionData(self.source, self.result()))
        self.assertFalse(self.config.output_dir.exists())

    def test_injected_predictor_contract(self):
        class FixedPredictor:
            def predict(self, image):
                return Prediction(np.full(image.shape[:2], 0.8, dtype=np.float32), "test predictor")
        result = extract(self.source, FixedPredictor(), 0.5, 0.5, "#2563EB", False)
        self.assertEqual(result.water_pixels, 9600)
        self.assertEqual(result.model_name, "test predictor")
        # 与训练指标保持相同的严格大于比较，概率等于阈值时属于背景。
        at_threshold = extract(self.source, FixedPredictor(), 0.8, 0.5, "#2563EB", False)
        self.assertEqual(at_threshold.water_pixels, 0)


if __name__ == "__main__":
    unittest.main()
