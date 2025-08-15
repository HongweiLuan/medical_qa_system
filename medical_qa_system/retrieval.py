from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text


def build_tfidf_index(answer_corpus: List[str]) -> Tuple[TfidfVectorizer, any]:
    """
    Fit a TF-IDF vectorizer on the answer corpus.

    Args:
        answer_corpus (List[str]): list of answer strings

    Returns:
        Tuple[TfidfVectorizer, sparse matrix]: vectorizer and TF-IDF matrix
    """
    vec = TfidfVectorizer()
    mat = vec.fit_transform(answer_corpus)
    return vec, mat


def build_bm25(answer_corpus: List[str]) -> BM25Okapi:
    """
    Build a BM25 ranking index over tokenized answers.

    Args:
        answer_corpus (List[str]): list of answer strings

    Returns:
        BM25Okapi: BM25 model
    """
    tokenized = [a.split() for a in answer_corpus]
    return BM25Okapi(tokenized)


def retrieve_with_tfidf(query: str, tfidf_vec: TfidfVectorizer, tfidf_matrix: any,
                        all_answers: List[str], top_k: int = 5) -> List[str]:
    """
    Retrieve top-k answers using TF-IDF cosine similarity.
    """
    q_vec = tfidf_vec.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [all_answers[i] for i in top_idx]


def retrieve_with_bm25(query: str, bm25_model: BM25Okapi, all_answers: List[str], top_k: int = 5) -> List[str]:
    """
    Retrieve top-k answers using BM25 ranking.
    """
    q_tokens = query.split()
    scores = bm25_model.get_scores(q_tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [all_answers[i] for i in top_idx]


def retrieve_with_dense(query: str, model: SentenceTransformer, answers_emb: np.ndarray, all_answers: List[str], top_k: int = 5, device: str = "cpu") -> List[Tuple[int, float, str]]:
    """Retrieve top-k answers using dense embedding dot-product similarity."""
    q_emb = model.encode([clean_text(query)], convert_to_numpy=True, normalize_embeddings=True, device=device)
    sims = np.dot(q_emb, answers_emb.T).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), all_answers[i]) for i in top_idx]
