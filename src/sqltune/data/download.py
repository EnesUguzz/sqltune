"""Download the official Spider zip (questions + SQLite databases) and unpack it.

Purpose: Hugging Face's Spider copy has no databases, and execution accuracy needs them;
the official zip on Google Drive has both.
Flow: `sqltune download` -> download_spider() -> <data_dir>/raw/spider.zip -> <data_dir>/spider/.
Params: DataConfig.data_dir, DataConfig.spider_drive_id.
Try: If gdown hits Drive's quota, download the zip in a browser from the Spider site,
save it as data/raw/spider.zip and re-run `sqltune download`.
"""

from __future__ import annotations

import shutil
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
BIRD_TRAIN_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"  # ~8.9 GB


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


def _sqlite_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    """(member, db_id) for every <db>/<db>.sqlite inside the zip; other files are ignored and
    names with '..' are rejected."""
    out = []
    for member in zf.infolist():
        parts = member.filename.split("/")
        if ".." in parts or not member.filename.endswith(".sqlite") or len(parts) < 2:
            continue
        db_id = parts[-2]
        if parts[-1] == f"{db_id}.sqlite":
            out.append((member, db_id))
    return out


def download_bird_train(cfg: DataConfig) -> Path:
    """Ensure <data_dir>/bird_train/ holds train.json and train_databases/<db>/<db>.sqlite from
    the official BIRD train zip (the databases sit in a zip inside the zip); returns that
    folder."""
    target = cfg.data_dir / "bird_train"
    raw = cfg.data_dir / "raw"
    zip_path = raw / "bird_train.zip"
    if (target / "train.json").exists() and (target / "train_databases").exists():
        return target
    if not zip_path.exists():
        _fetch_url(BIRD_TRAIN_ZIP_URL, zip_path)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as outer:
        names = outer.namelist()
        questions = next(n for n in names if n.split("/")[-1] == "train.json")
        (target / "train.json").write_bytes(outer.read(questions))
        inner_name = next((n for n in names if n.endswith("train_databases.zip")), None)
        inner_path = raw / "bird_train_databases.zip"
        if inner_name is not None and not inner_path.exists():
            with outer.open(inner_name) as src, inner_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 24)
        sources = [zipfile.ZipFile(inner_path)] if inner_name is not None else [outer]
        for zf in sources:
            for member, db_id in _sqlite_members(zf):
                out = target / "train_databases" / db_id / f"{db_id}.sqlite"
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 24)
    return target
