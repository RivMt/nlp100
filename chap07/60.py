PROBLEM_DESCRIPTION = "60. Download SST-2 and count positive/negative examples."

from collections import Counter

from chap07.common import load_split
from utils.path_solver import output_path


def main() -> None:
    result = {}
    for split in ("train", "dev"):
        counts = Counter(row["label"] for row in load_split(split))
        result[split] = {"positive": counts["1"], "negative": counts["0"]}
    lines = [
        f"{split}: positive (1) = {v['positive']}, negative (0) = {v['negative']}"
        for split, v in result.items()
    ]
    text = "\n".join(lines) + "\n"
    output_path(__file__, "label_counts.txt").write_text(text, encoding="utf-8")
    print(text, end="")

