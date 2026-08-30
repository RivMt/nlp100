PROBLEM_DESCRIPTION = "52. Find the ten words most similar to United_States."

from chap06.common import get_vector, load_vectors, most_similar_vector, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    rows = [
        {"word": word, "similarity": float(score)}
        for word, score in most_similar_vector(
            vectors,
            get_vector(vectors, "United_States"),
            10,
            exclude_terms=("United_States",),
        )
    ]
    save_json(output_path(__file__, "most_similar.json"), rows)
    for row in rows:
        print(row)

