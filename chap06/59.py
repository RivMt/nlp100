PROBLEM_DESCRIPTION = "59. Visualize country vectors with t-SNE."

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from chap06.common import countries_and_vectors, load_vectors
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    countries, matrix = countries_and_vectors(vectors)
    perplexity = min(30.0, max(2.0, (len(countries) - 1) / 3))
    points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    ).fit_transform(matrix)
    plt.figure(figsize=(14, 10))
    plt.scatter(points[:, 0], points[:, 1], s=18)
    for country, (x, y) in zip(countries, points):
        plt.annotate(country, (x, y), fontsize=8)
    plt.title("t-SNE of country word vectors")
    plt.tight_layout()
    destination = output_path(__file__, "tsne.png")
    plt.savefig(destination, dpi=180)
    print(f"saved={destination}")

