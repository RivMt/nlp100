"""Resolve resource and output paths consistently across exercise scripts."""

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_ROOT = PROJECT_ROOT / "res"
OUT_ROOT = PROJECT_ROOT / "out"


def resource_path(*parts: PathLike) -> Path:
    """Return a path below the project's ``res`` directory."""
    return RES_ROOT.joinpath(*parts)


def script_name(script_path: PathLike) -> str:
    """Return a script's filename without its extension."""
    return Path(script_path).stem


def output_dir(script_path: PathLike) -> Path:
    """Return and create ``out/{script_name}`` for the given script."""
    directory = OUT_ROOT / script_name(script_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_path(script_path: PathLike, *parts: PathLike) -> Path:
    """Return a path below ``out/{script_name}``, creating its parent directory."""
    path = output_dir(script_path).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
