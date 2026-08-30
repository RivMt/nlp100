PROBLEM_DESCRIPTION = "55. Measure semantic and syntactic analogy accuracy."

import csv
from collections import defaultdict

from chap06.common import save_json
from utils.path_solver import output_path
from utils.problem_dependencies import ensure_problem_outputs


def main() -> None:
    (source,) = ensure_problem_outputs(54, "capital_common_countries.tsv")
    counts = defaultdict(lambda: {"correct": 0, "total": 0, "oov": 0})
    with source.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            category = "syntactic" if row["section"].startswith("gram") else "semantic"
            counts[category]["total"] += 1
            if row["predicted"] == "<OOV>":
                counts[category]["oov"] += 1
            counts[category]["correct"] += int(row["correct"])
    result = {}
    for category in ("semantic", "syntactic"):
        row = counts[category]
        result[category] = {
            **row,
            "accuracy": row["correct"] / row["total"] if row["total"] else None,
        }
    save_json(output_path(__file__, "analogy_accuracy.json"), result)
    print(result)

