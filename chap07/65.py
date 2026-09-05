PROBLEM_DESCRIPTION = "65. Predict sentiment for an arbitrary text."

import argparse

from chap07.common import ensure_model, save_json, transform_texts
from utils.path_solver import output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", default=["the worst movie I 've ever seen"])
    args = parser.parse_args()
    text = " ".join(args.text)
    vectorizer, model = ensure_model()
    x = transform_texts(vectorizer, [text])
    prediction = int(model.predict(x)[0])
    probability_by_label = dict(zip(model.classes_, model.predict_proba(x)[0]))
    result = {
        "text": text,
        "label": prediction,
        "sentiment": "positive" if prediction else "negative",
        "probability": float(probability_by_label[prediction]),
    }
    save_json(output_path(__file__, "prediction.json"), result)
    print(result)

