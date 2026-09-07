PROBLEM_DESCRIPTION = "82. Show the top-10 [MASK] predictions and probabilities."

from chap09.common import HF_HOME, MODEL_NAME, device, save_json, tokenizer
from utils.path_solver import output_path


def main() -> None:
    import torch
    from transformers import AutoModelForMaskedLM

    text = "The movie was full of [MASK]."
    tok = tokenizer()
    target_device = device()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=HF_HOME).to(target_device)
    model.eval()
    inputs = tok(text, return_tensors="pt").to(target_device)
    mask = inputs.input_ids.eq(tok.mask_token_id).nonzero(as_tuple=False)[0]
    with torch.no_grad():
        probabilities = model(**inputs).logits[mask[0], mask[1]].softmax(-1)
    values, ids = probabilities.topk(10)
    result = [
        {"token": tok.convert_ids_to_tokens(int(i)), "probability": float(p)}
        for p, i in zip(values, ids)
    ]
    save_json(output_path(__file__, "top10.json"), result)
    for row in result:
        print(row)

