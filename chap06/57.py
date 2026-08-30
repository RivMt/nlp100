PROBLEM_DESCRIPTION = "57. Cluster country vectors into five groups with k-means."

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pycountry
from sklearn.cluster import KMeans

from chap06.common import WordVectors, load_vectors, resolve_key, save_json
from utils.path_solver import output_path


def country_vectors(vectors: WordVectors) -> Tuple[List[str], np.ndarray]:
    countries = []
    rows = []
    for country in pycountry.countries:
        names = [
            getattr(country, "common_name", ""),
            country.name,
            getattr(country, "official_name", ""),
        ]
        key = None
        for name in names:
            if name:
                key = resolve_key(vectors, name)
            if key is not None:
                break
        if key is None:
            continue
        countries.append(country.name)
        rows.append(vectors.vectors[vectors.word_to_index[key]])
    if len(countries) < 5:
        raise ValueError("Fewer than five country vectors are available")
    return countries, np.stack(rows)


def main() -> None:
    vectors = load_vectors()
    countries, matrix = country_vectors(vectors)
    labels = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(matrix)
    clusters = defaultdict(list)
    for country, label in zip(countries, labels):
        clusters[str(int(label))].append(country)
    result = dict(sorted(clusters.items()))
    save_json(output_path(__file__, "clusters.json"), result)
    for cluster, members in result.items():
        print(cluster, members)

