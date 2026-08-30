PROBLEM_DESCRIPTION = "50. Load Google News vectors and display United_States."

from chap06.common import MODEL_NAME, get_vector, load_vectors, resolve_key, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    vector = get_vector(vectors, "United_States")
    result = {
        "model": MODEL_NAME,
        "term": "United_States",
        "resolved_key": resolve_key(vectors, "United_States"),
        "dimensions": int(vector.shape[0]),
        "vector": vector.tolist(),
    }
    save_json(output_path(__file__, "united_states_vector.json"), result)
    print(result)

