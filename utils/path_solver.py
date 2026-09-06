from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_ROOT = PROJECT_ROOT / "res"
OUT_ROOT = PROJECT_ROOT / "out"


def resource_path(*parts: PathLike) -> Path:
    return RES_ROOT.joinpath(*parts)


def script_name(script_path: PathLike) -> str:
    return Path(script_path).stem


def output_dir(script_path: PathLike) -> Path:
    directory = OUT_ROOT / script_name(script_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_path(script_path: PathLike, *parts: PathLike) -> Path:
    path = output_dir(script_path).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
