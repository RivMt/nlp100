PROBLEM_DESCRIPTION = "77. Train the mini-batch model on a GPU."

import torch

from chap08.common import datasets, make_bow_model, save_training, train


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for problem 77")
    matrix, _, _, train_set, dev_set = datasets()
    model = make_bow_model(matrix, freeze=True)
    history = train(model, train_set, dev_set, device="cuda", batch_size=64)
    print(save_training(77, model.cpu(), history, {"freeze": True, "device": "cuda"}))

