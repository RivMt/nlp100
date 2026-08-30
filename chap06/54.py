PROBLEM_DESCRIPTION = "54. Predict capital/common-country analogies."

import csv

from chap06.common import analogy_examples, load_vectors, predict_analogy
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    destination = output_path(__file__, "capital_common_countries.tsv")
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            ["section", "first", "second", "third", "expected", "predicted", "similarity", "correct"]
        )
        for section, first, second, third, expected in analogy_examples(
            {"capital-common-countries"}
        ):
            try:
                predicted, similarity = predict_analogy(vectors, first, second, third)
            except KeyError:
                predicted, similarity = "<OOV>", float("nan")
            correct = predicted.lower() == expected.lower()
            writer.writerow(
                [section, first, second, third, expected, predicted, similarity, int(correct)]
            )
            print(first, second, third, expected, predicted, similarity)
    print(f"saved={destination}")

