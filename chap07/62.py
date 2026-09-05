PROBLEM_DESCRIPTION = "62. Train a logistic-regression classifier."

import joblib

from chap07.common import train_model
from utils.path_solver import output_path


def main() -> None:
    vectorizer, model = train_model()
    vectorizer_path = output_path(__file__, "vectorizer.joblib")
    model_path = output_path(__file__, "model.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    print(f"vocabulary_size={len(vectorizer.vocabulary_)}")
    print(f"saved={model_path}")

