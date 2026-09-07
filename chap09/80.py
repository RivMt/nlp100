PROBLEM_DESCRIPTION = "80. Tokenize the given sentence with a BERT tokenizer."

from chap09.common import save_json, tokenizer
from utils.path_solver import output_path


def main() -> None:
    text = "The movie was full of incomprehensibilities."
    tokens = tokenizer().tokenize(text)
    result = {"text": text, "tokens": tokens}
    save_json(output_path(__file__, "tokens.json"), result)
    print(tokens)

