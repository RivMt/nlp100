PROBLEM_DESCRIPTION = "71. Convert SST-2 examples to embedding token IDs."

from chap08.common import datasets, save_json
from utils.path_solver import output_path


def serializable(example):
    return {
        "text": example["text"],
        "label": example["label"].tolist(),
        "input_ids": example["input_ids"].tolist(),
    }


def main() -> None:
    _, _, _, train, dev = datasets()
    result = {
        "train_examples": len(train),
        "dev_examples": len(dev),
        "first_train_example": serializable(train[0]),
    }
    save_json(output_path(__file__, "dataset.json"), result)
    print(result)

