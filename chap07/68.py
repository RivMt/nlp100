PROBLEM_DESCRIPTION = "68. Inspect the 20 largest and smallest feature weights."

import csv

from chap07.common import ensure_model
from utils.path_solver import output_path


def main() -> None:
    vectorizer, model = ensure_model()
    names = vectorizer.get_feature_names_out()
    weighted = sorted(zip(names, model.coef_[0]), key=lambda item: item[1])
    rows = [("lowest", *item) for item in weighted[:20]]
    rows += [("highest", *item) for item in reversed(weighted[-20:])]
    path = output_path(__file__, "feature_weights.csv")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "feature", "weight"])
        writer.writerows(rows)
    for row in rows:
        printable_feature = ascii(row[1])[1:-1]
        print(f"{row[0]:7s} {printable_feature:24s} {row[2]: .6f}")

