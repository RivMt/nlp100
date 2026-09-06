from __future__ import annotations


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

