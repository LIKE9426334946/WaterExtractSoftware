"""读取本地保存结果；编号持久化，刷新或重启后保持不变。"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)
# 当前应用为单进程服务；保存与扫描共用锁，避免读到正在写入的记录。
history_lock = RLock()


@dataclass(frozen=True)
class SavedResult:
    number: int
    folder: Path
    metadata: dict

    @property
    def original(self) -> Path | None:
        path = self.folder / "prediction_original.png"
        return path if path.is_file() else None

    @property
    def mask(self) -> Path:
        return self.folder / "prediction_mask.png"

    @property
    def overlay(self) -> Path:
        return self.folder / "prediction_overlay.png"

    @property
    def downloads(self) -> list[str]:
        paths = [self.mask, self.overlay, self.folder / "prediction_results.zip"]
        return [str(path) for path in paths if path.is_file()]


def list_results(output_dir: Path) -> list[SavedResult]:
    """兼容旧保存目录；跳过未完成/损坏的记录，不读取训练输出。"""
    with history_lock:
        index_path = output_dir / "history.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        changed = False
        records = []
        required = {"source_filename", "created_at", "width", "height", "water_pixels",
                    "background_pixels", "water_percent", "elapsed_ms"}
        for stats_path in sorted(output_dir.glob("prediction_*/statistics.json")):
            folder = stats_path.parent
            try:
                metadata = json.loads(stats_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not required.issubset(metadata):
                    continue
                if not all((folder / name).is_file() for name in ("prediction_mask.png", "prediction_overlay.png")):
                    continue
            except (OSError, ValueError):
                logger.warning("跳过无法读取的保存记录：%s", folder)
                continue
            if folder.name not in index:
                index[folder.name] = max(index.values(), default=0) + 1
                changed = True
            records.append(SavedResult(index[folder.name], folder, metadata))
        if changed:
            temporary = index_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(index_path)
        return sorted(records, key=lambda record: record.number)
