"""
run_pipeline.py

This module provides:
1. Example interactions to demonstrate retrieval results for sample queries.
2. A main pipeline that runs the full workflow:
   - Data loading and preprocessing
   - Lexical baselines (TF-IDF & BM25)
   - Dense embeddings baseline
   - Fine-tuning a dense retriever
   - Model evaluation
"""

from typing import Dict
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from preprocessing import load_and_prepare, clean_text
from retrieval import build_tfidf_index, build_bm25, retrieve_with_dense
from training import build_training_examples, finetune_dense, embed_answers
from evaluation import evaluate_model, evaluate_bm25_retriever, evaluate_tfidf_retriever


# ---------------------------
# Example interactions
# ---------------------------
def print_examples(baseline_model, baseline_emb, ft_model, ft_emb, all_answers):
    """
    Prints top-3 retrieval results for a few sample queries using both
    baseline (pretrained) and fine-tuned dense models.

    Args:
        baseline_model: pretrained SentenceTransformer model
        baseline_emb: embeddings for baseline model
        ft_model: fine-tuned SentenceTransformer model
        ft_emb: embeddings for fine-tuned model
        all_answers: list of answer strings
    """
    examples = [
        "What are the symptoms of diabetes?",
        "How can I treat a common cold?",
        "What causes high blood pressure?"
    ]

    print("\n--- Example Interactions (Baseline) ---")
    for q in examples:
        ranked = retrieve_with_dense(q, baseline_model, baseline_emb, all_answers, top_k=3)
        print(f"\nUser: {q}")
        for i, (idx, score, ans) in enumerate(ranked, start=1):
            print(f"  {i}. score={score:.3f} ans={ans}")

    print("\n--- Example Interactions (Fine-tuned) ---")
    for q in examples:
        ranked = retrieve_with_dense(q, ft_model, ft_emb, all_answers, top_k=3)
        print(f"\nUser: {q}")
        for i, (idx, score, ans) in enumerate(ranked, start=1):
            print(f"  {i}. score={score:.3f} ans={ans}")


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(data_path: str = "mle_screening_dataset.csv",
                 base_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 output_dir: str = "dense_finetuned_model",
                 seed: int = 42,
                 eval_ks=[1, 3, 5, 10],
                 batch_size: int = 32) -> Dict:
    """
    Executes the full medical QA retrieval workflow:
      1. Load and preprocess the data
      2. Split into train/test sets
      3. Build TF-IDF and BM25 lexical baselines
      4. Compute baseline dense embeddings
      5. Prepare training examples and fine-tune dense retriever
      6. Evaluate all models
      7. Print example queries for demonstration

    Args:
        data_path (str): path to CSV dataset with columns 'question' and 'answer'
        base_model_name (str): pretrained SentenceTransformer model name
        output_dir (str): path to save fine-tuned model
        seed (int): random seed for reproducibility
        eval_ks (list): top-K values for evaluation
        batch_size (int): batch size for embedding computation

    Returns:
        dict: all objects for further analysis (models, embeddings, datasets, etc.)
    """

    # ---------------------------
    # Load and prepare data
    # ---------------------------
    print("Loading and preparing data...")
    df, unique_answers = load_and_prepare(data_path)
    print(f"Pairs: {len(df)}, Unique answers: {len(unique_answers)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    all_answers = unique_answers

    # ---------------------------
    # Build TF-IDF and BM25 baselines
    # ---------------------------
    tfidf_vec, tfidf_mat = build_tfidf_index(all_answers)
    bm25_model = build_bm25(all_answers)
    print("Built TF-IDF and BM25 indices.")

    print("\nEvaluating TF-IDF retriever...")
    for k in eval_ks:
        tfidf_metrics = evaluate_tfidf_retriever(test_df, tfidf_vec, tfidf_mat, all_answers, top_k=k)
        print(f"TF-IDF Recall@{k}: {tfidf_metrics[f'Recall@{k}']:.4f}, MRR@{k}: {tfidf_metrics[f'MRR@{k}']:.4f}")
    print("\nEvaluating BM25 retriever...")
    for k in eval_ks:
        bm25_metrics = evaluate_bm25_retriever(test_df, bm25_model, all_answers, top_k=k)
        print(f"BM25 Recall@{k}: {bm25_metrics[f'Recall@{k}']:.4f}, MRR@{k}: {bm25_metrics[f'MRR@{k}']:.4f}")

    # ---------------------------
    # Compute baseline dense embeddings
    # ---------------------------
    print("Loading baseline dense model and encoding corpus (this may download the model)...")
    baseline_model = SentenceTransformer(base_model_name)
    answers_emb_baseline = embed_answers(baseline_model, all_answers)
    print("Baseline embeddings done.")

    # ---------------------------
    # Prepare training examples and fine-tune dense retriever
    # ---------------------------
    train_examples = build_training_examples(train_df)
    print(f"Prepared {len(train_examples)} training examples.")

    print("Fine-tuning the dense retriever (this may take time)...")
    finetuned_model = finetune_dense(base_model_name, train_examples, output_dir)
    answers_emb_ft = embed_answers(finetuned_model, all_answers)
    print("Fine-tuned model saved and encoded answers.")

    # ---------------------------
    # Evaluate baseline vs fine-tuned models
    # ---------------------------
    print("\nEvaluating BASELINE model...")
    baseline_metrics = evaluate_model(baseline_model, answers_emb_baseline, test_df, all_answers, ks=eval_ks)
    for k in eval_ks:
        print(f"Baseline Recall@{k}: {baseline_metrics.get(f'Recall@{k}', 0):.4f}, "
              f"nDCG@{k}: {baseline_metrics.get(f'nDCG@{k}', 0):.4f}")
    print(f"Baseline MRR: {baseline_metrics['MRR']:.4f}, MAP: {baseline_metrics['MAP']:.4f}")

    print("\nEvaluating FINE-TUNED model...")
    finetuned_metrics = evaluate_model(finetuned_model, answers_emb_ft, test_df, all_answers, ks=eval_ks)
    for k in eval_ks:
        print(f"Finetuned Recall@{k}: {finetuned_metrics.get(f'Recall@{k}', 0):.4f}, "
              f"nDCG@{k}: {finetuned_metrics.get(f'nDCG@{k}', 0):.4f}")
    print(f"Finetuned MRR: {finetuned_metrics['MRR']:.4f}, MAP: {finetuned_metrics['MAP']:.4f}")

    # ---------------------------
    # Print example interactions
    # ---------------------------
    print_examples(baseline_model, answers_emb_baseline, finetuned_model, answers_emb_ft, all_answers)

    # ---------------------------
    # Return all relevant objects
    # ---------------------------
    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "tfidf_vec": tfidf_vec,
        "tfidf_mat": tfidf_mat,
        "bm25_model": bm25_model,
        "baseline_model": baseline_model,
        "answers_emb_baseline": answers_emb_baseline,
        "finetuned_model": finetuned_model,
        "answers_emb_ft": answers_emb_ft,
        "unique_answers": unique_answers
    }


# ---------------------------
# Script execution
# ---------------------------
if __name__ == "__main__":
    run_pipeline()
