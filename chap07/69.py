PROBLEM_DESCRIPTION = "69. Sweep regularization strength and plot dev accuracy."

import csv

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from chap07.common import load_feature_split, train_model
from utils.path_solver import output_path


def main() -> None:
    dev = load_feature_split("dev")
    gold = [int(row["label"]) for row in dev]
    strengths = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    scores = []
    for c in strengths:
        vectorizer, model = train_model(c=c)
        pred = model.predict(vectorizer.transform(row["feature"] for row in dev))
        score = float(accuracy_score(gold, pred))
        scores.append(score)
        print(f"C={c:g} dev_accuracy={score:.6f}")

    with output_path(__file__, "regularization.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["C", "dev_accuracy"])
        writer.writerows(zip(strengths, scores))

    plt.figure(figsize=(7, 4))
    plt.semilogx(strengths, scores, marker="o")
    plt.xlabel("Inverse regularization strength (C)")
    plt.ylabel("Development accuracy")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path(__file__, "regularization.png"), dpi=160)

