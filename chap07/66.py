PROBLEM_DESCRIPTION = "66. Build a confusion matrix on the development set."

from sklearn.metrics import confusion_matrix

from chap07.common import ensure_model, load_feature_split, save_json
from utils.path_solver import output_path


def main() -> None:
    vectorizer, model = ensure_model()
    rows = load_feature_split("dev")
    gold = [int(row["label"]) for row in rows]
    pred = model.predict(vectorizer.transform(row["feature"] for row in rows))
    matrix = confusion_matrix(gold, pred, labels=[0, 1]).tolist()
    result = {"labels": [0, 1], "matrix": matrix}
    save_json(output_path(__file__, "confusion_matrix.json"), result)
    print("rows=gold, columns=predicted; labels=[0, 1]")
    print(matrix)

