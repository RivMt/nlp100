PROBLEM_DESCRIPTION = "88. Predict sentiment using the model fine-tuned in problem 87."

from chap09.common import device, save_json
from utils.path_solver import output_path
from utils.problem_dependencies import ensure_problem_outputs


SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    (model_dir,) = ensure_problem_outputs(87, "model")
    target_device = device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(target_device)
    model.eval()
    inputs = tok(SENTENCES, padding=True, return_tensors="pt").to(target_device)
    with torch.no_grad():
        probabilities = model(**inputs).logits.softmax(-1).cpu()
    result = [
        {
            "text": text,
            "label": int(probability.argmax()),
            "positive_probability": float(probability[1]),
        }
        for text, probability in zip(SENTENCES, probabilities)
    ]
    save_json(output_path(__file__, "predictions.json"), result)
    for row in result:
        print(row)

