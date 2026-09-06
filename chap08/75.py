PROBLEM_DESCRIPTION = "75. Pad and length-sort a group of examples."

from chap08.common import collate, datasets, save_json
from utils.path_solver import output_path


def main() -> None:
    _, _, _, train, _ = datasets()
    batch = collate(train[:4])
    result = {
        "input_ids": batch["input_ids"].tolist(),
        "label": batch["label"].tolist(),
        "text": batch["text"],
    }
    save_json(output_path(__file__, "batch.json"), result)
    print(batch["input_ids"])
    print(batch["label"])

