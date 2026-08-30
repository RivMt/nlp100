"""Environment-variable helpers shared by exercise chapters."""

from __future__ import annotations

import os


def env_int(name: str, default: int = 0) -> int:
    """Read an integer environment variable with a clear validation error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error

