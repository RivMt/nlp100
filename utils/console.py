"""Console compatibility helpers for exercise scripts."""

from __future__ import annotations

import sys


def configure_console() -> None:
    """Avoid crashes when a Windows legacy code page cannot encode model tokens."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")

