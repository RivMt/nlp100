PROBLEM_DESCRIPTION = "56. Evaluate vectors on WordSimilarity-353 with Spearman correlation."

import csv

from scipy.stats import spearmanr

from chap06.common import cosine_similarity, ensure_wordsim353, get_vector, load_vectors, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    rows = []
    skipped = 0
    with ensure_wordsim353().open(encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            word_a, word_b, human = line.rstrip().split("\t")
            try:
                model_score = cosine_similarity(
                    get_vector(vectors, word_a), get_vector(vectors, word_b)
                )
            except KeyError:
                skipped += 1
                continue
            rows.append((word_a, word_b, float(human), model_score))

    correlation = spearmanr(
        [row[2] for row in rows], [row[3] for row in rows]
    ).statistic # pyright: ignore[reportAttributeAccessIssue]
    with output_path(__file__, "word_similarity_predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["word_a", "word_b", "human_score", "model_similarity"])
        writer.writerows(rows)
    result = {"pairs": len(rows), "skipped_oov": skipped, "spearman_correlation": float(correlation)}
    save_json(output_path(__file__, "metrics.json"), result)
    print(result)

