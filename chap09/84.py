PROBLEM_DESCRIPTION = "84. Compare mean-pooled final-layer sentence vectors."

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
        hidden = model(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1)
    vectors = ((hidden * mask).sum(1) / mask.sum(1)).cpu()
    result = cosine_pairs(vectors)
    save_json(output_path(__file__, "similarities.json"), result)
    for row in result:
        print(row)

