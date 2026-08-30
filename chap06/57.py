PROBLEM_DESCRIPTION = "57. Cluster country vectors into five groups with k-means."

from collections import defaultdict

from sklearn.cluster import KMeans

from chap06.common import countries_and_vectors, load_vectors, save_json
from utils.path_solver import output_path


def main() -> None:
    vectors = load_vectors()
    countries, matrix = countries_and_vectors(vectors)
    labels = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(matrix)
    clusters = defaultdict(list)
    for country, label in zip(countries, labels):
        clusters[str(int(label))].append(country)
    result = dict(sorted(clusters.items()))
    save_json(output_path(__file__, "clusters.json"), result)
    for cluster, members in result.items():
        print(cluster, members)

