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


BIRD_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
BIRD_QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/"
    "mini_dev_sqlite-00000-of-00001.json"
)
BIRD_DB_PREFIX = "minidev/MINIDEV/dev_databases/"


def _fetch_url(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)


def download_bird(cfg: DataConfig) -> Path:
    """Ensure <data_dir>/bird/ holds Mini-Dev: questions from Hugging Face (the canonical,
    updated copy) and the SQLite databases from the official zip; returns that folder."""
    import json

    target = cfg.data_dir / "bird"
    questions = target / "mini_dev_sqlite.json"
    raw = cfg.data_dir / "raw"
    if not questions.exists():
        src = raw / "bird_mini_dev_sqlite.json"
        if not src.exists():
            _fetch_url(BIRD_QUESTIONS_URL, src)
        text = src.read_text(encoding="utf-8").strip()
        records = (
            json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines()]
        )
        target.mkdir(parents=True, exist_ok=True)
        questions.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    if not (target / "dev_databases").exists():
        zip_path = raw / "minidev.zip"
        if not zip_path.exists():
            _fetch_url(BIRD_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                name = member.filename
                if name.startswith(BIRD_DB_PREFIX) and name.endswith(".sqlite"):
                    out = target / "dev_databases" / name[len(BIRD_DB_PREFIX) :]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(member))
    return target
