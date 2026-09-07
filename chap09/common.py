from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import configure_huggingface_cache
from utils.console import configure_console
from utils.env import env_int
from utils.files import save_json
from utils.path_solver import OUT_ROOT
from utils.problem_dependencies import ensure_problem_outputs
from utils.sst2 import load_sst2_split
from utils.torch_utils import device

configure_console()

HF_HOME = configure_huggingface_cache()

MODEL_NAME = os.environ.get("NLP100_BERT_MODEL", "bert-base-uncased")
MAX_LENGTH = env_int("NLP100_MAX_LENGTH", 128)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)


def encoded_sst2(split: str, tok=None) -> List[Dict[str, Any]]:
    tok = tok or tokenizer()
    result = []
    for row in load_sst2_split(split, limit_from_environment=True):
        encoded = tok(row["sentence"], truncation=True, max_length=MAX_LENGTH)
        result.append(
            {
                "text": row["sentence"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": int(row["label"]),
            }
        )
    return result


def model_inputs(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels": x["labels"],
        }
        for x in examples
    ]


def load_tokenized_sst2() -> Dict[str, List[Dict[str, Any]]]:
    import torch

    (dataset_path,) = ensure_problem_outputs(85, "tokenized_sst2.pt")
    return torch.load(dataset_path, map_location="cpu", weights_only=False)


def evaluate_classifier(model, examples, tok, target_device: str) -> float:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    loader = DataLoader(
        model_inputs(examples),
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tok),
    )
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(target_device)
            batch = {k: v.to(target_device) for k, v in batch.items()}
            pred = model(**batch).logits.argmax(-1)
            correct += pred.eq(labels).sum().item()
            total += labels.numel()
    return correct / total


def train_classifier(problem: int = 87) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
    )

    tok = tokenizer()
    datasets = load_tokenized_sst2()
    train_set = datasets["train"]
    dev_set = datasets["dev"]
    target_device = device()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir=HF_HOME
    ).to(target_device)
    loader = DataLoader(
        model_inputs(train_set),
        batch_size=env_int("NLP100_BATCH_SIZE", 16),
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tok),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = env_int("NLP100_EPOCHS", 2)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            batch = {k: v.to(target_device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        accuracy = evaluate_classifier(model, dev_set, tok, target_device)
        record = {"epoch": epoch, "loss": loss_sum / len(loader), "dev_accuracy": accuracy}
        history.append(record)
        print(record)

    model_dir = OUT_ROOT / str(problem) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    result = {
        "model": MODEL_NAME,
        "device": target_device,
        "train_examples": len(train_set),
        "dev_examples": len(dev_set),
        "history": history,
    }
    save_json(OUT_ROOT / str(problem) / "metrics.json", result)
    return result


SENTENCES = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def cosine_pairs(vectors) -> List[Dict[str, Any]]:
    import torch.nn.functional as functional

    rows = []
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            rows.append(
                {
                    "sentence_a": SENTENCES[i],
                    "sentence_b": SENTENCES[j],
                    "cosine_similarity": float(
                        functional.cosine_similarity(vectors[i], vectors[j], dim=0)
                    ),
                }
            )
    return rows
