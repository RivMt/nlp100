PROBLEM_DESCRIPTION = "63. Predict the first development example."

from chap07.common import ensure_model, load_feature_split, save_json
from utils.path_solver import output_path


def main() -> None:
    vectorizer, model = ensure_model()
    row = load_feature_split("dev")[0]
    predicted = int(model.predict(vectorizer.transform([row["feature"]]))[0])
    result = {
        "text": row["text"],
        "gold": int(row["label"]),
        "predicted": predicted,
        "correct": predicted == int(row["label"]),
    }
    save_json(output_path(__file__, "prediction.json"), result)
    print(result)

