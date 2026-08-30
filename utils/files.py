"""Small file serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, value: Any) -> None:
    """Write UTF-8, human-readable JSON and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

