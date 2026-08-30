"""Download and load the SST-2 files shared by chapters 7–10."""

from __future__ import annotations

import csv
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

from utils.env import env_int
from utils.path_solver import resource_path


SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
SST2_DIR = resource_path("SST-2")


def ensure_sst2() -> Path:
    """Download only the labeled train and development files when missing."""
    required = ("train.tsv", "dev.tsv")
    if all((SST2_DIR / name).is_file() for name in required):
        return SST2_DIR
    SST2_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "SST-2.zip"
        urllib.request.urlretrieve(SST2_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            available = set(archive.namelist())
            for name in required:
                member = f"SST-2/{name}"
                if member not in available:
                    raise FileNotFoundError(f"{member} is missing from the archive")
                with archive.open(member) as src, (SST2_DIR / name).open("wb") as dst:
                    dst.write(src.read())
    return SST2_DIR


def load_sst2_split(
    split: str,
    *,
    limit_from_environment: bool = False,
) -> List[Dict[str, str]]:
    """Load train/dev rows, optionally respecting NLP100_MAX_EXAMPLES."""
    if split not in {"train", "dev"}:
        raise ValueError(f"Unsupported SST-2 split: {split!r}")
    path = ensure_sst2() / f"{split}.tsv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows or not {"sentence", "label"}.issubset(rows[0]):
        raise ValueError(f"Unexpected SST-2 format: {path}")
    if limit_from_environment:
        limit = env_int("NLP100_MAX_EXAMPLES")
        if limit:
            rows = rows[:limit]
    return rows
