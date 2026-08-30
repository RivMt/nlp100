"""Run prerequisite problems only when their expected outputs are missing."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, List, Set

from utils.path_solver import OUT_ROOT, PathLike


_RUNNING_PROBLEMS: Set[int] = set()


def _problem_main(problem: int) -> Callable[[], object]:
    chapter = problem // 10 + 1
    module_name = f"chap{chapter:02d}.{problem}"
    module = importlib.import_module(module_name)
    callback = getattr(module, "main", None)
    if not callable(callback):
        raise TypeError(f"{module_name} has no callable main")
    return callback


def ensure_problem_outputs(problem: int, *relative_paths: PathLike) -> List[Path]:
    """Return prerequisite outputs, running its problem once when any are absent."""
    outputs = [OUT_ROOT / str(problem) / Path(path) for path in relative_paths]
    missing = [path for path in outputs if not path.exists()]
    if not missing:
        return outputs
    if problem in _RUNNING_PROBLEMS:
        raise RuntimeError(f"Cyclic problem dependency detected at problem {problem}")

    print(
        f"Problem {problem} output is missing; running prerequisite: "
        + ", ".join(str(path) for path in missing),
        flush=True,
    )
    _RUNNING_PROBLEMS.add(problem)
    try:
        _problem_main(problem)()
    finally:
        _RUNNING_PROBLEMS.remove(problem)

    missing = [path for path in outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Problem {problem} did not create expected output(s): "
            + ", ".join(str(path) for path in missing)
        )
    return outputs
