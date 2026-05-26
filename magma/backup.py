"""Backup utilities for MAGMA local storage."""

import json
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from magma.graph.sqlite_store import DB_PATH


def _data_dir(db_path: str) -> Path:
    return Path(db_path).resolve().parent


def _backup_dir(db_path: str, backup_dir: Optional[str]) -> Path:
    if backup_dir:
        return Path(backup_dir).resolve()
    return _data_dir(db_path) / "backups"


def _checkpoint_wal(db_path: str) -> Dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return {
            "busy": int(row[0] or 0),
            "log_frames": int(row[1] or 0),
            "checkpointed_frames": int(row[2] or 0),
        }
    finally:
        conn.close()


def _prune_backups(directory: Path, keep_days: int, keep_latest: int) -> int:
    archives = sorted(directory.glob("magma-backup-*.zip"), key=lambda item: item.stat().st_mtime, reverse=True)
    cutoff = datetime.utcnow() - timedelta(days=max(keep_days, 1))
    removed = 0
    for index, archive in enumerate(archives):
        modified = datetime.utcfromtimestamp(archive.stat().st_mtime)
        if index < keep_latest or modified >= cutoff:
            continue
        archive.unlink()
        removed += 1
    return removed


def create_backup(
    db_path: str = DB_PATH,
    backup_dir: Optional[str] = None,
    keep_days: int = 14,
    keep_latest: int = 7,
) -> Dict[str, object]:
    db = Path(db_path).resolve()
    if not db.exists():
        raise FileNotFoundError(f"MAGMA database not found: {db}")

    directory = _backup_dir(str(db), backup_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_wal(str(db))
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    archive = directory / f"magma-backup-{timestamp}.zip"
    data_dir = _data_dir(str(db))
    include_names = [
        db.name,
        "faiss.index",
        "faiss_meta.json",
        "id_map.json",
    ]
    included = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = data_dir / name
            if path.exists():
                zf.write(path, arcname=name)
                included.append(name)
        metadata = {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "db_path": str(db),
            "included": included,
            "checkpoint": checkpoint,
            "pid": os.getpid(),
        }
        zf.writestr("backup-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

    removed = _prune_backups(directory, keep_days=keep_days, keep_latest=keep_latest)
    return {
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "included": included,
        "checkpoint": checkpoint,
        "pruned": removed,
        "keep_days": keep_days,
        "keep_latest": keep_latest,
    }
