"""通过真实本地 HTTP 服务验证 Gradio 序列化、会话和下载，不需浏览器。"""

import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import httpx
import numpy as np
from gradio_client import Client, handle_file
from PIL import Image

from models import MODEL_NAMES, create_predictor
from ui.layout import build_app, build_theme
from utils.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


class GradioRoundtripTests(unittest.TestCase):
    def test_upload_extract_restyle_save_invalidate_replace_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = replace(load_config(ROOT / "config.yaml"), output_dir=directory / "outputs")
            app = build_app(cfg, create_predictor())
            _, url, _ = app.launch(
                server_name="127.0.0.1", share=False, prevent_thread_lock=True,
                theme=build_theme(), css_paths=ROOT / "ui" / "styles.css",
                footer_links=[], quiet=True,
            )
            try:
                with httpx.Client(trust_env=False) as http:
                    self.assertEqual(http.get(url).status_code, 200)
                    config = http.get(url + "config").json()
                    self.assertIn("遥感影像水体智能提取系统", config["title"])
                    self.assertEqual(config["version"], "6.30.0")
                client = Client(url, verbose=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    empty = client.predict(None, "U-Net", 0.5, 0.5, "#2563EB", False, api_name="/extract")
                self.assertIsNone(empty[2])
                self.assertIn("请先上传", empty[7])

                path = directory / "test.PNG"
                rgb = np.full((96, 144, 3), [80, 120, 100], dtype=np.uint8)
                Image.fromarray(rgb).save(path)
                upload = handle_file(str(path))
                loaded = client.predict(upload, api_name="/load_image")
                self.assertIn("144 × 96", loaded[4])
                self.assertIsNone(loaded[2])
                for name in MODEL_NAMES:
                    extracted = client.predict(upload, name, 0.5, 0.5, "#2563EB", False, api_name="/extract")
                    self.assertIn(name, extracted[6])
                    self.assertIn("模拟提取完成", extracted[7])
                with Image.open(extracted[2]) as mask_image:
                    mask = np.asarray(mask_image).copy()
                    self.assertEqual(mask_image.size, (144, 96))
                    self.assertEqual(set(np.unique(mask)), {0, 255})

                styled = client.predict(upload, 1, "#FF0000", False, api_name="/restyle")
                with Image.open(styled[3]) as overlay:
                    pixels = np.asarray(overlay)
                    self.assertTrue(np.all(pixels[mask != 0] == [255, 0, 0]))
                    np.testing.assert_array_equal(pixels[mask == 0], rgb[mask == 0])
                saved = client.predict(upload, "SegFormer", 0.5, 1, "#FF0000", False, api_name="/save_results")
                self.assertIn("保存成功", saved[7])
                files = saved[8]["value"]
                self.assertEqual(len(files), 3)
                for file in files:
                    self.assertTrue(Path(file).is_file())
                with ZipFile(files[2]) as archive:
                    meta = json.loads(archive.read("statistics.json"))
                    self.assertEqual(meta["color"], "#FF0000")
                    self.assertEqual(meta["water_pixels"], int(np.count_nonzero(mask)))
                    self.assertTrue(meta["is_simulation"])

                # 新会话不能看到或保存另一个会话的结果。
                other = Client(url, verbose=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    isolated = other.predict(upload, "SegFormer", 0.5, 1, "#FF0000", False, api_name="/save_results")
                self.assertFalse(isolated[8]["visible"])
                self.assertIn("请先上传", isolated[7])
                invalidated = client.predict(api_name="/invalidate")
                self.assertIsNone(invalidated[2])
                self.assertFalse(invalidated[8]["visible"])
                full = client.predict(upload, "U-Net", 0, 0.5, "#2563EB", False, api_name="/extract")
                with Image.open(full[2]) as full_mask:
                    self.assertTrue(np.all(np.asarray(full_mask) == 255))
                empty_mask = client.predict(upload, "U-Net", 1, 0.5, "#2563EB", False, api_name="/extract")
                with Image.open(empty_mask[2]) as zeros:
                    self.assertTrue(np.all(np.asarray(zeros) == 0))

                second = directory / "second.jpeg"
                Image.new("RGB", (100, 60)).save(second)
                replaced = client.predict(handle_file(str(second)), api_name="/load_image")
                self.assertIsNone(replaced[2])
                self.assertIn("100 × 60", replaced[4])
                corrupt = directory / "corrupt.png"
                corrupt.write_bytes(b"not an image")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    rejected = client.predict(handle_file(str(corrupt)), api_name="/load_image")
                self.assertIsNone(rejected[0])
                self.assertIn("无法读取", rejected[7])
                cleared = client.predict(api_name="/clear")
                self.assertTrue(all(item is None for item in cleared[:5]))
                self.assertIn("已清空", cleared[8])
                self.assertFalse(cleared[9]["visible"])
                # 清空只影响会话，不删除此前显式保存的文件。
                self.assertEqual(len(list(cfg.output_dir.glob("*/simulated_results.zip"))), 1)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
