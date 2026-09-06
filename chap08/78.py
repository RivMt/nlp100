PROBLEM_DESCRIPTION = "78. Fine-tune the embedding matrix together with the classifier."

import torch

from chap08.common import datasets, make_bow_model, save_training, train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix, _, _, train_set, dev_set = datasets()
    model = make_bow_model(matrix, freeze=False)
    history = train(
        model, train_set, dev_set, device=device, batch_size=64, learning_rate=1e-4
    )
    print(save_training(78, model.cpu(), history, {"freeze": False, "device": device}))

