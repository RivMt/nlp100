PROBLEM_DESCRIPTION = "61. Convert SST-2 text into whitespace-tokenized BoW dictionaries."

import json
from typing import Dict

from chap07.common import load_split, text_to_feature
from utils.path_solver import output_path


def convert(row: Dict[str, str]) -> Dict[str, object]:
    return {
        "text": row["sentence"],
        "label": row["label"],
        "feature": text_to_feature(row["sentence"]),
    }


def main() -> None:
    first = None
    for split in ("train", "dev"):
        path = output_path(__file__, f"{split}_features.jsonl")
        with path.open("w", encoding="utf-8") as file:
            for row in load_split(split):
                example = convert(row)
                first = first or example
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps(first, ensure_ascii=False, indent=2))

