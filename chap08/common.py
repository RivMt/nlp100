"""Shared PyTorch data/model code for chapter 8."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import configure_gensim_cache
from utils.console import configure_console
from utils.env import env_int
from utils.files import save_json
from utils.path_solver import OUT_ROOT, resource_path
from utils.problem_dependencies import ensure_problem_outputs
from utils.sst2 import load_sst2_split

configure_console()

configure_gensim_cache()

EMBEDDING_NAME = os.environ.get("NLP100_EMBEDDING", "glove-wiki-gigaword-50")
EMBEDDING_DIR = resource_path("embeddings")
SST2_DIR = resource_path("SST-2")


def embedding_cache_path(name: str = EMBEDDING_NAME) -> Path:
    return EMBEDDING_DIR / f"{name}.pt"


def load_embeddings(name: str = EMBEDDING_NAME):
    """Return (matrix, word_to_id, id_to_word), reserving ID 0 for PAD."""
    import torch

    cache = embedding_cache_path(name)
    if cache.is_file():
        return torch.load(cache, map_location="cpu", weights_only=False)

    import gensim.downloader as api

    vectors = api.load(name)
    words = vectors.index_to_key # pyright: ignore[reportAttributeAccessIssue]
    matrix = torch.zeros((len(words) + 1, vectors.vector_size), dtype=torch.float32) # pyright: ignore[reportAttributeAccessIssue]
    matrix[1:] = torch.from_numpy(vectors.vectors.copy()) # pyright: ignore[reportAttributeAccessIssue]
    word_to_id = {word: i + 1 for i, word in enumerate(words)}
    id_to_word = ["<PAD>", *words]
    bundle = (matrix, word_to_id, id_to_word)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache)
    return bundle


def encode_rows(split: str, word_to_id: Dict[str, int]):
    import torch

    examples = []
    rows = load_sst2_split(split, limit_from_environment=True)
    for row in rows:
        ids = [word_to_id[token] for token in row["sentence"].split() if token in word_to_id]
        if ids:
            examples.append(
                {
                    "text": row["sentence"],
                    "label": torch.tensor([float(row["label"])]),
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                }
            )
    return examples


def collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    ordered = sorted(examples, key=lambda x: len(x["input_ids"]), reverse=True)
    return {
        "input_ids": pad_sequence(
            [x["input_ids"] for x in ordered], batch_first=True, padding_value=0
        ),
        "label": torch.stack([x["label"] for x in ordered]),
        "text": [x["text"] for x in ordered],
    }


def make_bow_model(matrix, freeze: bool = True):
    import torch

    class AverageBoW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                matrix, freeze=freeze, padding_idx=0
            )
            self.output = torch.nn.Linear(matrix.shape[1], 1)

        def forward(self, input_ids):
            mask = input_ids.ne(0).unsqueeze(-1)
            summed = (self.embedding(input_ids) * mask).sum(dim=1)
            mean = summed / mask.sum(dim=1).clamp_min(1)
            return self.output(mean)

    return AverageBoW()


def make_mlp_model(matrix, freeze: bool = False):
    import torch

    class MLPBoW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                matrix, freeze=freeze, padding_idx=0
            )
            dim = matrix.shape[1]
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.25),
                torch.nn.Linear(128, 1),
            )

        def forward(self, input_ids):
            mask = input_ids.ne(0).unsqueeze(-1)
            mean = (self.embedding(input_ids) * mask).sum(1)
            mean = mean / mask.sum(1).clamp_min(1)
            return self.classifier(mean)

    return MLPBoW()


def evaluate(model, examples, device="cpu", batch_size: int = 256) -> float:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device))
            labels = batch["label"].to(device)
            correct += ((logits >= 0) == labels.bool()).sum().item()
            total += labels.numel()
    return correct / total


def train(
    model,
    train_examples,
    dev_examples,
    *,
    device="cpu",
    batch_size: int = 64,
    epochs: Optional[int] = None,
    learning_rate: float = 1e-3,
):
    import torch
    from torch.utils.data import DataLoader

    epochs = epochs or env_int("NLP100_EPOCHS", 5)
    model.to(device)
    loader = DataLoader(
        train_examples, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch["label"])
        dev_accuracy = evaluate(model, dev_examples, device=device)
        record = {
            "epoch": epoch,
            "loss": loss_sum / len(train_examples),
            "dev_accuracy": dev_accuracy,
        }
        history.append(record)
        print(record)
    return history


def datasets():
    matrix, word_to_id, id_to_word = load_embeddings()
    return matrix, word_to_id, id_to_word, encode_rows(
        "train", word_to_id
    ), encode_rows("dev", word_to_id)


def save_training(problem: int, model, history, metadata: Dict[str, Any]) -> Path:
    import torch

    directory = OUT_ROOT / str(problem)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    torch.save({"state_dict": model.state_dict(), **metadata}, checkpoint)
    save_json(directory / "history.json", history)
    return checkpoint


def load_problem_model(problem: int, matrix, model_factory=make_bow_model):
    import torch

    (checkpoint,) = ensure_problem_outputs(problem, "model.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_factory(matrix, freeze=payload.get("freeze", True))
    model.load_state_dict(payload["state_dict"])
    return model
