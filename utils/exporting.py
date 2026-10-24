"""保存原尺寸 PNG 与统计信息，提供单文件和 ZIP 下载。"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, PngImagePlugin

from utils.history import history_lock, list_results
from utils.pipeline import ExtractionResult


def save_result(result: ExtractionResult, output_dir: Path) -> list[str]:
    with history_lock:
        return _write_result(result, output_dir)


def _write_result(result: ExtractionResult, output_dir: Path) -> list[str]:
    prefix = "prediction"
    folder = output_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:10]}"
    folder.mkdir(parents=True, exist_ok=False)
    notice = "水体分割模型预测结果。"
    metadata = {
        "software": "WaterExtraction", "is_simulation": False,
        "notice": notice, "created_at": datetime.now().astimezone().isoformat(),
        "source_filename": result.source.filename,
        "width": result.source.width, "height": result.source.height,
        "spatial_resolution": None, "model": result.model_name,
        "threshold": result.threshold,
        "alpha": result.alpha, "color": result.color, "show_boundary": result.show_boundary,
        "water_pixels": result.water_pixels, "background_pixels": result.background_pixels,
        "total_pixels": result.total_pixels, "water_percent": result.water_percent,
        "elapsed_ms": result.elapsed_ms,
        "timing_scope": "predict + score validation + thresholding; excludes visualization and file IO",
        "mask_values": {"background": 0, "water": 255},
    }
    png_info = PngImagePlugin.PngInfo()
    png_info.add_itxt("Notice", notice)
    png_info.add_text("Simulation", "false")
    png_info.add_text("Software", "WaterExtraction")
    original_path = folder / f"{prefix}_original.png"
    mask_path = folder / f"{prefix}_mask.png"
    overlay_path = folder / f"{prefix}_overlay.png"
    stats_path = folder / "statistics.json"
    archive_path = folder / f"{prefix}_results.zip"
    try:
        # 保存处理后的 RGB 原图，历史查看不依赖上传缓存的生命周期。
        Image.fromarray(result.source.rgb).save(original_path)
        Image.fromarray(result.mask).save(mask_path, pnginfo=png_info)
        Image.fromarray(result.overlay).save(overlay_path, pnginfo=png_info)
        stats_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            for path in (original_path, mask_path, overlay_path, stats_path):
                archive.write(path, arcname=path.name)
            archive.writestr("说明.txt", notice + "\n掩码：0=背景，255=水体。面积占比为像素占比，不是地理面积。\n")
        list_results(output_dir)
    except Exception:
        # 仅删除本次新建的未完成目录，不触碰既有保存结果。
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return [str(mask_path.resolve()), str(overlay_path.resolve()), str(archive_path.resolve())]
