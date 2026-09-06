PROBLEM_DESCRIPTION = "72. Construct an average-embedding Bag-of-Words model."

from chap08.common import load_embeddings, make_bow_model
from utils.path_solver import output_path


def main() -> None:
    matrix, _, _ = load_embeddings()
    model = make_bow_model(matrix, freeze=True)
    text = str(model) + "\n"
    output_path(__file__, "model.txt").write_text(text, encoding="utf-8")
    print(text, end="")

