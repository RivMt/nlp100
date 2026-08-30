PROBLEM_DESCRIPTION = "50. Load Google News vectors and display United_States."

from chap06.common import get_vector, load_vectors, resolve_key, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    vector = get_vector(vectors, "United_States")
    result = {
        "model": "GoogleNews-vectors-negative300",
        "term": "United_States",
        "resolved_key": resolve_key(vectors, "United_States"),
        "dimensions": int(vector.shape[0]),
        "vector": vector.tolist(),
    }
    save_json(output_path(__file__, "united_states_vector.json"), result)
    print(result)

