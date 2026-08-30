"""Run one NLP 100 exercise by problem number."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Tuple

from utils.console import configure_console


PROJECT_ROOT = Path(__file__).resolve().parent
MIN_PROBLEM = 50
MAX_PROBLEM = 99

configure_console()


def problem_script(problem: int) -> Path:
    if not MIN_PROBLEM <= problem <= MAX_PROBLEM:
        raise ValueError(
            f"problem must be between {MIN_PROBLEM} and {MAX_PROBLEM}, got {problem}"
        )
    chapter = problem // 10 + 1
    script = PROJECT_ROOT / f"chap{chapter:02d}" / f"{problem}.py"
    if not script.is_file():
        raise FileNotFoundError(f"Problem script does not exist: {script}")
    return script


def _load_problem_module(problem: int) -> ModuleType:
    """Import a numeric problem module."""
    chapter = problem // 10 + 1
    return importlib.import_module(f"chap{chapter:02d}.{problem}")


def _build_problem_registry() -> Tuple[
    Dict[str, Callable[[], object]],
    Dict[str, str],
]:
    functions: Dict[str, Callable[[], object]] = {}
    descriptions: Dict[str, str] = {}
    for chapter in range(6, 11):
        for problem in range((chapter - 1) * 10, chapter * 10):
            script = PROJECT_ROOT / f"chap{chapter:02d}" / f"{problem}.py"
            if not script.is_file():
                continue
            module = _load_problem_module(problem)
            callback = getattr(module, "main", None)
            description = getattr(module, "PROBLEM_DESCRIPTION", None)
            if not callable(callback):
                raise TypeError(f"chap{chapter:02d}/{problem}.py has no callable main")
            if not isinstance(description, str) or not description:
                raise TypeError(
                    f"chap{chapter:02d}/{problem}.py has no PROBLEM_DESCRIPTION"
                )
            key = str(problem)
            functions[key] = callback
            descriptions[key] = description
    return functions, descriptions


PROBLEM_FUNCTIONS, PROBLEM_DESCRIPTIONS = _build_problem_registry()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an NLP 100 exercise by problem number."
    )
    parser.add_argument("problem", type=int, help="problem number (50-99)")
    parser.add_argument(
        "problem_args",
        nargs=argparse.REMAINDER,
        help="additional arguments forwarded to the problem script",
    )
    args = parser.parse_args()
    try:
        script = problem_script(args.problem)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    key = str(args.problem)
    relative = script.relative_to(PROJECT_ROOT)
    print(PROBLEM_DESCRIPTIONS[key], flush=True)
    print(f"Running problem {args.problem}: {relative}", flush=True)
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    try:
        sys.argv = [str(script), *args.problem_args]
        os.chdir(PROJECT_ROOT)
        result = PROBLEM_FUNCTIONS[key]()
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
