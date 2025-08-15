from typing import List, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np

def build_training_examples(train_df) -> List[InputExample]:
    """
    Convert dataframe into a list of InputExample for fine-tuning.
    """
    examples = []
    for _, row in train_df.iterrows():
        q = row["question_clean"]
        a_pos = row["answer_clean"]
        examples.append(InputExample(texts=[q, a_pos]))
    return examples

def finetune_dense(base_model_name: str,
                   train_examples: List[InputExample],
                   output_dir: str,
                   batch_size: int = 32,
                   epochs: int = 3,
                   warmup_steps: int = 200,
                   device: str = "cpu") -> SentenceTransformer:
    """
    Fine-tune a SentenceTransformer using MultipleNegativesRankingLoss.
    """
    model = SentenceTransformer(base_model_name, device=device)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True
    )
    return model

def embed_answers(model: SentenceTransformer, answers: List[str], batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """
    Compute dense embeddings for answers.
    """
    return model.encode(
        answers,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device
    )

def retrieve_with_dense(query: str, model: SentenceTransformer, answers_emb: np.ndarray, all_answers: List[str], top_k: int = 5, device: str = "cpu") -> List[Tuple[int, float, str]]:
    """Retrieve top-k answers using dense embedding dot-product similarity."""
    q_emb = model.encode([clean_text(query)], convert_to_numpy=True, normalize_embeddings=True, device=device)
    sims = np.dot(q_emb, answers_emb.T).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), all_answers[i]) for i in top_idx]
