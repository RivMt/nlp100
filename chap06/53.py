PROBLEM_DESCRIPTION = "53. Solve Spain - Madrid + Athens by vector arithmetic."

from chap06.common import get_vector, load_vectors, most_similar_vector, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    query = get_vector(vectors, "Spain") - get_vector(vectors, "Madrid") + get_vector(vectors, "Athens")
    rows = [
        {"word": word, "similarity": float(score)}
        for word, score in most_similar_vector(
            vectors, query, 10, exclude_terms=("Spain", "Madrid", "Athens")
        )
    ]
    save_json(output_path(__file__, "analogy.json"), rows)
    for row in rows:
        print(row)

