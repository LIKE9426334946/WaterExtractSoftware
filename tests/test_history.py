"""保存快照、持久编号及历史详情回归检查。"""

import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import gradio as gr
import numpy as np
from PIL import Image

from models import Prediction
from ui.history import HistoryController
from utils.exporting import save_result
from utils.history import list_results
from utils.image_processing import load_image
from utils.pipeline import extract


class HistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "outputs"
        self.path = self.root / "source.png"
        Image.new("RGB", (48, 32), (30, 80, 100)).save(self.path)
        source = load_image(self.path, 20, 1000000)

        class Predictor:
            def predict(self, image):
                return Prediction(np.full(image.shape[:2], 0.8, dtype=np.float32), "test")

        self.result = extract(source, Predictor(), 0.5, 0.5, "#2563EB", False)

    def test_snapshot_survives_upload_deletion_and_has_original_in_archive(self):
        files = save_result(self.result, self.output)
        self.path.unlink()
        record = list_results(self.output)[0]
        self.assertEqual(record.number, 1)
        with Image.open(record.original) as original:
            np.testing.assert_array_equal(np.asarray(original), self.result.source.rgb)
        with ZipFile(files[-1]) as archive:
            self.assertIn("prediction_original.png", archive.namelist())
        controller = HistoryController(self.output)
        records, gallery, summary = controller.refresh()
        self.assertEqual(gallery["value"][0][1], "编号 1")
        self.assertIn("1 条", summary)
        detail = controller.open(records, gr.SelectData(None, {"index": 0, "value": None}))
        self.assertEqual(detail[:3], (str(record.original), str(record.mask), str(record.overlay)))
        self.assertIn("100.00%", detail[3])
        self.assertIn("编号 1", detail[4])
        self.assertEqual(detail[5], files)

    def test_numbers_survive_new_controllers_and_removed_records_are_not_reused(self):
        for _ in range(2):
            save_result(self.result, self.output)
        first = HistoryController(self.output).refresh()[0]
        self.assertEqual([record.number for record in first], [1, 2])
        shutil.rmtree(first[1].folder)
        save_result(self.result, self.output)
        reloaded = HistoryController(self.output).refresh()[0]
        self.assertEqual([record.number for record in reloaded], [1, 3])
        self.assertEqual(first[0].folder, reloaded[0].folder)

    def test_old_records_without_original_are_imported_with_explicit_notice(self):
        save_result(self.result, self.output)
        record = list_results(self.output)[0]
        record.original.unlink()
        (self.output / "history.json").unlink()  # 模拟升级前已存在的保存目录。
        controller = HistoryController(self.output)
        records, gallery, _ = controller.refresh()
        self.assertEqual(gallery["value"], [(str(record.overlay), "编号 1")])
        detail = controller.open(records, gr.SelectData(None, {"index": 0, "value": None}))
        self.assertIsNone(detail[0])
        self.assertIn("旧记录未保存原始影像", detail[4])
        self.assertTrue(Path(detail[1]).is_file())

    def test_concurrent_saves_have_distinct_stable_numbers(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: save_result(self.result, self.output), range(3)))
        records = list_results(self.output)
        self.assertEqual([record.number for record in records], [1, 2, 3])
        self.assertEqual(len({record.folder for record in records}), 3)
        self.assertEqual(len(json.loads((self.output / "history.json").read_text())), 3)

    def test_empty_incomplete_and_failed_saves_do_not_create_history_entries(self):
        records, gallery, summary = HistoryController(self.output).refresh()
        self.assertEqual(records, [])
        self.assertEqual(gallery["value"], [])
        self.assertIn("暂无保存结果", summary)
        save_result(self.result, self.output)
        before = (self.output / "history.json").read_bytes()
        with patch("utils.exporting.Image.Image.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_result(self.result, self.output)
        incomplete = self.output / "prediction_incomplete"
        incomplete.mkdir()
        (incomplete / "statistics.json").write_text("{}")
        self.assertEqual(len(list_results(self.output)), 1)
        self.assertEqual((self.output / "history.json").read_bytes(), before)

    def test_history_keeps_saved_style_and_escapes_filename(self):
        result = replace(self.result, source=replace(self.result.source, filename='<img src=x onerror="alert(1)">.png'))
        save_result(result, self.output)
        controller = HistoryController(self.output)
        records = controller.refresh()[0]
        detail = controller.open(records, gr.SelectData(None, {"index": 0, "value": None}))
        with Image.open(detail[2]) as overlay:
            np.testing.assert_array_equal(np.asarray(overlay), result.overlay)
        self.assertNotIn("<img", detail[4])
        self.assertIn("&lt;img", detail[4])


if __name__ == "__main__":
    unittest.main()
