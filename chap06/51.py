PROBLEM_DESCRIPTION = "51. Compute cosine similarity between United_States and U.S."

from chap06.common import cosine_similarity, get_vector, load_vectors, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    similarity = cosine_similarity(
        get_vector(vectors, "United_States"), get_vector(vectors, "U.S.")
    )
    result = {"word_a": "United_States", "word_b": "U.S.", "cosine_similarity": similarity}
    save_json(output_path(__file__, "similarity.json"), result)
    print(result)

