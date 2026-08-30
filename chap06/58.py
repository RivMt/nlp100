PROBLEM_DESCRIPTION = "58. Perform Ward hierarchical clustering and draw a dendrogram."

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from chap06.common import countries_and_vectors, load_vectors
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    countries, matrix = countries_and_vectors(vectors)
    hierarchy = linkage(matrix, method="ward")
    plt.figure(figsize=(16, 8))
    dendrogram(hierarchy, labels=countries, leaf_rotation=90, leaf_font_size=8)
    plt.title("Ward clustering of country word vectors")
    plt.ylabel("Distance")
    plt.tight_layout()
    destination = output_path(__file__, "dendrogram.png")
    plt.savefig(destination, dpi=180)
    print(f"saved={destination}")

