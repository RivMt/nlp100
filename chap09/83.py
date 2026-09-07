PROBLEM_DESCRIPTION = "83. Compare sentence vectors represented by the final [CLS] state."

from chap09.common import HF_HOME, MODEL_NAME, SENTENCES, cosine_pairs, device, save_json, tokenizer
from utils.path_solver import output_path


def main() -> None:
    import torch
    from transformers import AutoModel

    tok = tokenizer()
    target_device = device()
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=HF_HOME).to(target_device)
    model.eval()
    inputs = tok(SENTENCES, padding=True, return_tensors="pt").to(target_device)
    with torch.no_grad():
        vectors = model(**inputs).last_hidden_state[:, 0].cpu()
    result = cosine_pairs(vectors)
    save_json(output_path(__file__, "similarities.json"), result)
    for row in result:
        print(row)

