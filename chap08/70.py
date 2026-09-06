PROBLEM_DESCRIPTION = "70. Load pretrained embeddings and reserve row 0 for PAD."

from chap08.common import EMBEDDING_NAME, embedding_cache_path, load_embeddings, save_json
from utils.path_solver import output_path


def main() -> None:
    matrix, word_to_id, id_to_word = load_embeddings()
    assert matrix[0].count_nonzero().item() == 0
    result = {
        "embedding": EMBEDDING_NAME,
        "shape": list(matrix.shape),
        "pad_id": 0,
        "first_token": id_to_word[1],
        "first_token_id": word_to_id[id_to_word[1]],
        "cache": str(embedding_cache_path()),
    }
    save_json(output_path(__file__, "embedding.json"), result)
    print(result)

