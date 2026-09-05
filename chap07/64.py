PROBLEM_DESCRIPTION = "64. Compute conditional label probabilities for the first dev example."

from chap07.common import ensure_model, load_feature_split, save_json
from utils.path_solver import output_path


def main() -> None:
    vectorizer, model = ensure_model()
    row = load_feature_split("dev")[0]
    probabilities = model.predict_proba(vectorizer.transform([row["feature"]]))[0]
    result = {str(label): float(p) for label, p in zip(model.classes_, probabilities)}
    save_json(output_path(__file__, "probabilities.json"), result)
    print(result)

