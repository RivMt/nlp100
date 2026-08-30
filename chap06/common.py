"""Shared word-vector and evaluation utilities for chapter 6."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import configure_gensim_cache
from utils.console import configure_console
from utils.env import env_int
from utils.files import save_json
from utils.path_solver import resource_path

configure_console()

configure_gensim_cache()

MODEL_NAME = os.environ.get("NLP100_WORD_VECTORS", "word2vec-google-news-300")
QUESTIONS_URL = "https://download.tensorflow.org/data/questions-words.txt"
QUESTIONS_PATH = resource_path("questions-words", "questions-words.txt")
WORDSIM_URL = (
    "https://raw.githubusercontent.com/piskvorky/gensim/develop/"
    "gensim/test/test_data/wordsim353.tsv"
)
WORDSIM_PATH = resource_path("WordSimilarity-353", "wordsim353.tsv")


def load_vectors():
    """Load vectors through gensim, whose cache is fixed below res/."""
    import gensim.downloader as api

    return api.load(MODEL_NAME)


def candidate_keys(term: str) -> List[str]:
    candidates = [term, term.lower()]
    if " " in term:
        candidates.extend([term.replace(" ", "_"), term.lower().replace(" ", "_")])
    return list(dict.fromkeys(candidates))


def resolve_key(vectors, term: str) -> Optional[str]:
    for candidate in candidate_keys(term):
        if candidate in vectors:
            return candidate
    return None


def get_vector(vectors, term: str) -> np.ndarray:
    """Resolve casing; for small smoke-test models, average phrase components."""
    key = resolve_key(vectors, term)
    if key is not None:
        return np.asarray(vectors[key])
    parts = term.replace("_", " ").split()
    resolved = [resolve_key(vectors, part) for part in parts]
    if len(parts) > 1 and all(resolved):
        return np.mean([vectors[key] for key in resolved], axis=0)
    raise KeyError(f"{term!r} is not covered by {MODEL_NAME}")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def most_similar_vector(
    vectors,
    vector: np.ndarray,
    topn: int = 10,
    exclude_terms: Tuple[str, ...] = (),
):
    restrict = env_int("NLP100_RESTRICT_VOCAB") or None
    excluded = set()
    for term in exclude_terms:
        excluded.update(candidate.lower() for candidate in candidate_keys(term))
        for part in term.replace("_", " ").split():
            excluded.update(candidate.lower() for candidate in candidate_keys(part))
    candidates = vectors.most_similar(
        vector,
        topn=topn + len(excluded) + 10,
        restrict_vocab=restrict,
    )
    return [item for item in candidates if item[0].lower() not in excluded][:topn]


def ensure_questions() -> Path:
    if not QUESTIONS_PATH.is_file():
        QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)
        except Exception:
            # The course URL occasionally presents a mismatched TLS certificate.
            # Gensim ships an identical copy, so keep certificate checks enabled
            # and use that trusted local copy instead.
            from gensim.test.utils import datapath

            shutil.copyfile(datapath("questions-words.txt"), QUESTIONS_PATH)
    return QUESTIONS_PATH


def ensure_wordsim353() -> Path:
    if WORDSIM_PATH.is_file():
        return WORDSIM_PATH
    WORDSIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(WORDSIM_URL, WORDSIM_PATH)
    except Exception:
        # Gensim ships the same benchmark as test data, providing an offline fallback.
        from gensim.test.utils import datapath

        shutil.copyfile(datapath("wordsim353.tsv"), WORDSIM_PATH)
    return WORDSIM_PATH


def analogy_examples(
    section_filter: Optional[Set[str]] = None,
) -> Iterator[Tuple[str, str, str, str, str]]:
    section = ""
    per_section: Dict[str, int] = {}
    limit = env_int("NLP100_ANALOGY_LIMIT")
    with ensure_questions().open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith(":"):
                section = line[1:].strip()
                continue
            if not line or (section_filter and section not in section_filter):
                continue
            if limit and per_section.get(section, 0) >= limit:
                continue
            fields = line.split()
            if len(fields) != 4:
                continue
            per_section[section] = per_section.get(section, 0) + 1
            yield section, fields[0], fields[1], fields[2], fields[3]


def predict_analogy(vectors, first: str, second: str, third: str):
    query = get_vector(vectors, second) - get_vector(vectors, first) + get_vector(vectors, third)
    candidates = most_similar_vector(
        vectors, query, topn=1, exclude_terms=(first, second, third)
    )
    if not candidates:
        raise RuntimeError("No analogy candidate remained after excluding input words")
    word, similarity = candidates[0]
    return word, float(similarity)


def countries_and_vectors(vectors):
    countries = []
    seen = set()
    for _, _, country_a, _, country_b in analogy_examples({"capital-common-countries"}):
        for country in (country_a, country_b):
            if country not in seen:
                try:
                    vector = get_vector(vectors, country)
                except KeyError:
                    continue
                seen.add(country)
                countries.append((country, vector))
    if len(countries) < 5:
        raise ValueError("Fewer than five country vectors are available")
    return [item[0] for item in countries], np.stack([item[1] for item in countries])
