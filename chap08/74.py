PROBLEM_DESCRIPTION = "74. Evaluate the problem 73 model on the development set."

from chap08.common import datasets, evaluate, load_problem_model, save_json
from utils.path_solver import output_path


def main() -> None:
    matrix, _, _, _, dev = datasets()
    model = load_problem_model(73, matrix)
    accuracy = evaluate(model, dev)
    result = {"dev_accuracy": accuracy}
    save_json(output_path(__file__, "metrics.json"), result)
    print(result)

