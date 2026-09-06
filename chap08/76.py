PROBLEM_DESCRIPTION = "76. Train the model with padded mini-batches."

from chap08.common import datasets, make_bow_model, save_training, train


def main() -> None:
    matrix, _, _, train_set, dev_set = datasets()
    model = make_bow_model(matrix, freeze=True)
    history = train(model, train_set, dev_set, batch_size=64, learning_rate=1e-3)
    print(save_training(76, model, history, {"freeze": True}))

