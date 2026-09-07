PROBLEM_DESCRIPTION = "89. Fine-tune a BERT classifier with max pooling instead of [CLS]."

from chap09.common import (
    HF_HOME,
    MODEL_NAME,
    device,
    load_tokenized_sst2,
    model_inputs,
    save_json,
    tokenizer,
)
from utils.env import env_int
from utils.path_solver import output_path


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, DataCollatorWithPadding

    class MaxPoolClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
            self.output = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask, labels=None):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            masked = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4)
            logits = self.output(masked.max(dim=1).values)
            loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
            return logits, loss

    tok = tokenizer()
    datasets = load_tokenized_sst2()
    train_set = datasets["train"]
    dev_set = datasets["dev"]
    collator = DataCollatorWithPadding(tok)
    target_device = device()
    model = MaxPoolClassifier().to(target_device)
    train_loader = DataLoader(
        model_inputs(train_set), batch_size=env_int("NLP100_BATCH_SIZE", 16),
        shuffle=True, collate_fn=collator
    )
    dev_loader = DataLoader(model_inputs(dev_set), batch_size=32, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    history = []
    for epoch in range(1, env_int("NLP100_EPOCHS", 2) + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(target_device) for k, v in batch.items()}
            optimizer.zero_grad()
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dev_loader:
                labels = batch.pop("labels").to(target_device)
                logits, _ = model(**{k: v.to(target_device) for k, v in batch.items()})
                correct += logits.argmax(-1).eq(labels).sum().item()
                total += len(labels)
        record = {"epoch": epoch, "loss": total_loss / len(train_loader), "dev_accuracy": correct / total}
        history.append(record)
        print(record)
    torch.save(model.cpu().state_dict(), output_path(__file__, "max_pool_model.pt"))
    save_json(
        output_path(__file__, "metrics.json"),
        {
            "model": MODEL_NAME,
            "pooling": "max",
            "device": target_device,
            "train_examples": len(train_set),
            "dev_examples": len(dev_set),
            "history": history,
        },
    )

