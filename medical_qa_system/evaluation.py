import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from retrieval import retrieve_with_bm25, retrieve_with_tfidf

def recall_at_k(sim_row: np.ndarray, gold_idx: int, k: int) -> int:
    """
    Check if the gold answer is in top-k results.
    """
    top = np.argpartition(-sim_row, k-1)[:k]
    return int(gold_idx in top)

def mrr_from_scores(sim_row: np.ndarray, gold_idx: int) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.
    """
    order = np.argsort(-sim_row)
    rank_positions = np.where(order == gold_idx)[0]
    if len(rank_positions) == 0:
        return 0.0
    rank = int(rank_positions[0]) + 1
    return 1.0 / rank

def dcg_at_k(rels: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.
    """
    rels = np.array(rels[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def ndcg_at_k(sim_row: np.ndarray, gold_idx: int, k: int) -> float:
    """
    Compute Normalized DCG at K.
    """
    order = np.argsort(-sim_row)[:k]
    rels = [1 if idx == gold_idx else 0 for idx in order]
    ideal_rels = [1] + [0]*(k-1)
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(sim_row: np.ndarray, gold_idx: int) -> float:
    """
    Compute average precision (1/rank of gold answer).
    """
    order = np.argsort(-sim_row)
    for r, idx in enumerate(order, start=1):
        if idx == gold_idx:
            return 1.0 / r
    return 0.0

def evaluate_model(model: SentenceTransformer, answers_emb: np.ndarray, test_df,
                   all_answers: List[str], ks: List[int] = [1,3,5,10], batch_size: int = 128,
                   device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate model with Recall@K, nDCG@K, MRR, and MAP.
    """
    ans2idx = {a: i for i, a in enumerate(all_answers)}
    questions = test_df["question_clean"].tolist()
    gold_answers = test_df["answer_clean"].tolist()
    gold_indices = [ans2idx[a] for a in gold_answers]
    q_embs = model.encode(questions, batch_size=batch_size, convert_to_numpy=True,
                          show_progress_bar=False, normalize_embeddings=True, device=device)
    sims = np.dot(q_embs, answers_emb.T)
    metrics = {}
    for k in ks:
        recs, ndcgs = [], []
        for i in range(len(questions)):
            sim_row = sims[i]
            gold_idx = gold_indices[i]
            recs.append(recall_at_k(sim_row, gold_idx, k))
            ndcgs.append(ndcg_at_k(sim_row, gold_idx, k))
        metrics[f"Recall@{k}"] = float(np.mean(recs))
        metrics[f"nDCG@{k}"] = float(np.mean(ndcgs))
    metrics["MRR"] = float(np.mean([mrr_from_scores(sims[i], gold_indices[i]) for i in range(len(questions))]))
    metrics["MAP"] = float(np.mean([average_precision(sims[i], gold_indices[i]) for i in range(len(questions))]))
    return metrics


def evaluate_tfidf_retriever(test_df: pd.DataFrame, tfidf_vec, tfidf_mat, all_answers: List[str], top_k: int = 5) -> Dict[str, float]:
    """Evaluate TF-IDF retriever for Recall@k and MRR."""
    ans2idx = {a: i for i, a in enumerate(all_answers)}
    recall_list = []
    mrr_list = []
    for _, row in test_df.iterrows():
        q = row["question_clean"]
        gold_idx = ans2idx[row["answer_clean"]]
        ranked = retrieve_with_tfidf(q, tfidf_vec, tfidf_mat, all_answers, top_k)
        ranked_indices = [ans2idx[a] for a in ranked]
        recall_list.append(int(gold_idx in ranked_indices))
        if gold_idx in ranked_indices:
            rank = ranked_indices.index(gold_idx) + 1
            mrr_list.append(1.0 / rank)
        else:
            mrr_list.append(0.0)
    return {"Recall@{}".format(top_k): float(np.mean(recall_list)), "MRR@{}".format(top_k): float(np.mean(mrr_list))}

def evaluate_bm25_retriever(test_df: pd.DataFrame, bm25_model, all_answers: List[str], top_k: int = 5) -> Dict[str, float]:
    """Evaluate BM25 retriever for Recall@k and MRR."""
    ans2idx = {a: i for i, a in enumerate(all_answers)}
    recall_list = []
    mrr_list = []
    for _, row in test_df.iterrows():
        q = row["question_clean"]
        gold_idx = ans2idx[row["answer_clean"]]
        ranked = retrieve_with_bm25(q, bm25_model, all_answers, top_k)
        ranked_indices = [ans2idx[a] for a in ranked]
        recall_list.append(int(gold_idx in ranked_indices))
        if gold_idx in ranked_indices:
            rank = ranked_indices.index(gold_idx) + 1
            mrr_list.append(1.0 / rank)
        else:
            mrr_list.append(0.0)
    return {"Recall@{}".format(top_k): float(np.mean(recall_list)), "MRR@{}".format(top_k): float(np.mean(mrr_list))}
