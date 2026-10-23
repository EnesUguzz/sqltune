"""Download the official Spider zip (questions + SQLite databases) and unpack it.

Purpose: Hugging Face's Spider copy has no databases, and execution accuracy needs them;
the official zip on Google Drive has both.
Flow: `sqltune download` -> download_spider() -> <data_dir>/raw/spider.zip -> <data_dir>/spider/.
Params: DataConfig.data_dir, DataConfig.spider_drive_id.
Try: If gdown hits Drive's quota, download the zip in a browser from the Spider site,
save it as data/raw/spider.zip and re-run `sqltune download`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from sqltune.config.models import DataConfig

ZIP_ROOT = "spider_data/"
REQUIRED = ("dev.json", "train_spider.json", "tables.json", "database")


def _fetch_zip(drive_id: str, dest: Path) -> None:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=drive_id, output=str(dest), quiet=False)


def _extract(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if not member.filename.startswith(ZIP_ROOT) or member.is_dir():
                continue  # skips __MACOSX/ metadata
            rel = member.filename[len(ZIP_ROOT) :]
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))


def download_spider(cfg: DataConfig) -> Path:
    """Ensure <data_dir>/spider/ holds the unpacked dataset; returns that folder."""
    target = cfg.data_dir / "spider"
    if (target / "dev.json").exists():
        return target
    zip_path = cfg.data_dir / "raw" / "spider.zip"
    if not zip_path.exists():
        _fetch_zip(cfg.spider_drive_id, zip_path)
    _extract(zip_path, target)
    missing = [name for name in REQUIRED if not (target / name).exists()]
    if missing:
        raise FileNotFoundError(f"Spider zip is missing {missing}; re-download {zip_path}")
    return target
