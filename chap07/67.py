PROBLEM_DESCRIPTION = "67. Measure accuracy, precision, recall and F1."

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict

from chap07.common import ensure_model, load_feature_split, save_json
from utils.path_solver import output_path


def metrics(gold, pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    vectorizer, model = ensure_model()
    result = {}
    for split in ("train", "dev"):
        rows = load_feature_split(split)
        gold = [int(row["label"]) for row in rows]
        pred = model.predict(vectorizer.transform(row["feature"] for row in rows))
        result[split] = metrics(gold, pred)
    save_json(output_path(__file__, "metrics.json"), result)
    print(result)

