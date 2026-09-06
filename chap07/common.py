"""Shared utilities for chapter 7 (classical machine learning)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.console import configure_console
from utils.files import save_json
from utils.problem_dependencies import ensure_problem_outputs
from utils.sst2 import load_sst2_split

configure_console()


def load_split(split: str) -> List[Dict[str, str]]:
    return load_sst2_split(split)


class FeatureExample(TypedDict):
    text: str
    label: str
    feature: Dict[str, int]


def text_to_feature(text: str) -> Dict[str, int]:
    return dict(Counter(text.split()))


def load_feature_split(split: str) -> List[FeatureExample]:
    if split not in {"train", "dev"}:
        raise ValueError(f"Unsupported feature split: {split!r}")
    (path,) = ensure_problem_outputs(61, f"{split}_features.jsonl")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_vectorizer():
    from sklearn.feature_extraction import DictVectorizer

    return DictVectorizer(sparse=True)


def transform_texts(vectorizer, texts):
    return vectorizer.transform(text_to_feature(text) for text in texts)


def train_model(c: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    train = load_feature_split("train")
    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(row["feature"] for row in train)
    y_train = [int(row["label"]) for row in train]
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    return vectorizer, model


def ensure_model():
    import joblib

    vectorizer_path, model_path = ensure_problem_outputs(
        62, "vectorizer.joblib", "model.joblib"
    )
    return joblib.load(vectorizer_path), joblib.load(model_path)
