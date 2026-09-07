PROBLEM_DESCRIPTION = "85. Load and tokenize the SST-2 train/development sets."

import torch

from chap09.common import encoded_sst2, save_json, tokenizer
from utils.path_solver import output_path


def main() -> None:
    tok = tokenizer()
    datasets = {split: encoded_sst2(split, tok) for split in ("train", "dev")}
    torch.save(datasets, output_path(__file__, "tokenized_sst2.pt"))
    summary = {
        split: {"examples": len(rows), "first": rows[0]} for split, rows in datasets.items()
    }
    save_json(output_path(__file__, "summary.json"), summary)
    print({split: len(rows) for split, rows in datasets.items()})

