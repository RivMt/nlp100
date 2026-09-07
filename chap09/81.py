PROBLEM_DESCRIPTION = "81. Predict the most likely token for [MASK]."

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
        logits = model(**inputs).logits[mask[0], mask[1]]
    token_id = int(logits.argmax())
    result = {"token": tok.convert_ids_to_tokens(token_id), "token_id": token_id}
    save_json(output_path(__file__, "prediction.json"), result)
    print(result)

