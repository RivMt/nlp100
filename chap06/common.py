"""Shared word-vector and evaluation utilities for chapter 6."""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.console import configure_console
from utils.env import env_int
from utils.files import save_json
from utils.path_solver import resource_path

configure_console()

DEFAULT_WORD2VEC_PATH = resource_path("GoogleNews-vectors-negative300.bin")
GZIP_WORD2VEC_PATH = resource_path(
    "gensim-data", "word2vec-google-news-300", "word2vec-google-news-300.gz"
)
QUESTIONS_URL = "https://download.tensorflow.org/data/questions-words.txt"
QUESTIONS_PATH = resource_path("questions-words", "questions-words.txt")
WORDSIM_URL = (
    "https://raw.githubusercontent.com/piskvorky/gensim/develop/"
    "gensim/test/test_data/wordsim353.tsv"
)
WORDSIM_PATH = resource_path("WordSimilarity-353", "wordsim353.tsv")


class WordVectors(NamedTuple):
    words: List[str]
    vectors: np.ndarray
    word_to_index: Dict[str, int]


PathLike = Union[str, Path]
_VECTOR_NORMS: Dict[int, np.ndarray] = {}


def _word2vec_path() -> Path:
    configured = os.environ.get("NLP100_WORD2VEC_PATH")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_WORD2VEC_PATH.is_file():
        return DEFAULT_WORD2VEC_PATH
    if GZIP_WORD2VEC_PATH.is_file():
        return GZIP_WORD2VEC_PATH
    return DEFAULT_WORD2VEC_PATH


def _read_binary_vector(file: BinaryIO, dim: int) -> np.ndarray:
    data = file.read(dim * np.dtype(np.float32).itemsize)
    if len(data) != dim * np.dtype(np.float32).itemsize:
        raise EOFError("Unexpected end of the word2vec binary file")
    return np.frombuffer(data, dtype=np.float32, count=dim)


def load_word2vec(path: PathLike) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Word2Vec file does not exist: {source}. "
            "Set NLP100_WORD2VEC_PATH to GoogleNews-vectors-negative300.bin."
        )
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rb") as file:
        vocab_size, dim = map(int, file.readline().split())
        words: List[str] = []
        vectors = np.empty((vocab_size, dim), dtype=np.float32)
        for index in range(vocab_size):
            chars = []
            while True:
                char = file.read(1)
                if not char:
                    raise EOFError("Unexpected end of the word2vec vocabulary")
                if char == b" ":
                    break
                if char != b"\n":
                    chars.append(char)
            words.append(b"".join(chars).decode("utf-8"))
            vectors[index] = _read_binary_vector(file, dim)
    word_to_index = {word: index for index, word in enumerate(words)}
    return words, vectors, word_to_index


def load_vectors() -> WordVectors:
    return WordVectors(*load_word2vec(_word2vec_path()))


def candidate_keys(term: str) -> List[str]:
    candidates = [term, term.lower()]
    if " " in term:
        candidates.extend([term.replace(" ", "_"), term.lower().replace(" ", "_")])
    return list(dict.fromkeys(candidates))


def resolve_key(vectors: WordVectors, term: str) -> Optional[str]:
    for candidate in candidate_keys(term):
        if candidate in vectors.word_to_index:
            return candidate
    return None


def get_vector(vectors: WordVectors, term: str) -> np.ndarray:
    """Resolve casing; for small smoke-test models, average phrase components."""
    key = resolve_key(vectors, term)
    if key is not None:
        return vectors.vectors[vectors.word_to_index[key]]
    parts = term.replace("_", " ").split()
    resolved = [resolve_key(vectors, part) for part in parts]
    if len(parts) > 1 and all(resolved):
        return np.mean(
            [vectors.vectors[vectors.word_to_index[key]] for key in resolved if key],
            axis=0,
        )
    raise KeyError(f"{term!r} is not covered by {_word2vec_path()}")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def most_similar_vector(
    vectors: WordVectors,
    vector: np.ndarray,
    topn: int = 10,
    exclude_terms: Tuple[str, ...] = (),
) -> List[Tuple[str, float]]:
    restrict = env_int("NLP100_RESTRICT_VOCAB")
    limit = min(restrict or len(vectors.words), len(vectors.words))
    excluded = set()
    for term in exclude_terms:
        excluded.update(candidate.lower() for candidate in candidate_keys(term))
        for part in term.replace("_", " ").split():
            excluded.update(candidate.lower() for candidate in candidate_keys(part))
    matrix = vectors.vectors[:limit]
    cache_key = id(vectors.vectors)
    norms = _VECTOR_NORMS.get(cache_key)
    if norms is None:
        norms = np.linalg.norm(vectors.vectors, axis=1)
        _VECTOR_NORMS[cache_key] = norms
    denominator = norms[:limit] * np.linalg.norm(vector)
    scores = np.divide(
        matrix @ vector,
        denominator,
        out=np.full(limit, -np.inf, dtype=np.float32),
        where=denominator != 0,
    )
    candidate_count = min(topn + len(excluded) + 10, limit)
    if candidate_count == 0:
        return []
    indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    candidates = [(vectors.words[index], float(scores[index])) for index in indices]
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


def predict_analogy(
    vectors: WordVectors, first: str, second: str, third: str
) -> Tuple[str, float]:
    query = get_vector(vectors, second) - get_vector(vectors, first) + get_vector(vectors, third)
    candidates = most_similar_vector(
        vectors, query, topn=1, exclude_terms=(first, second, third)
    )
    if not candidates:
        raise RuntimeError("No analogy candidate remained after excluding input words")
    word, similarity = candidates[0]
    return word, float(similarity)


def countries_and_vectors(vectors: WordVectors) -> Tuple[List[str], np.ndarray]:
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
