PROBLEM_DESCRIPTION = "79. Replace the linear head with a multi-layer neural network."

import torch

from chap08.common import datasets, make_mlp_model, save_training, train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix, _, _, train_set, dev_set = datasets()
    model = make_mlp_model(matrix, freeze=False)
    history = train(
        model, train_set, dev_set, device=device, batch_size=64, learning_rate=1e-4
    )
    print(
        save_training(
            79, model.cpu(), history, {"freeze": False, "device": device, "model": "MLPBoW"}
        )
    )

