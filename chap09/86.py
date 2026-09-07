PROBLEM_DESCRIPTION = "86. Create a padded mini-batch from four SST-2 examples."

from chap09.common import load_tokenized_sst2, model_inputs, save_json, tokenizer
from utils.path_solver import output_path


def main() -> None:
    from transformers import DataCollatorWithPadding

    tok = tokenizer()
    examples = load_tokenized_sst2()["train"][:4]
    batch = DataCollatorWithPadding(tok)(model_inputs(examples))
    result = {key: value.tolist() for key, value in batch.items()}
    save_json(output_path(__file__, "batch.json"), result)
    for key, value in batch.items():
        print(key, value)

