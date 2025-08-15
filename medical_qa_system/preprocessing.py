import string
import pandas as pd
from typing import List, Tuple


def clean_text(s: str) -> str:
    """
    Clean and normalize text by:
    - Converting to lowercase
    - Removing punctuation
    - Collapsing multiple spaces

    Args:
        s (str): input text

    Returns:
        str: cleaned text
    """
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def load_and_prepare(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load dataset, clean text, remove duplicates, and assign canonical answer IDs.

    Args:
        data_path (str): path to CSV with columns "question" and "answer"

    Returns:
        Tuple[pd.DataFrame, List[str]]: processed DataFrame and list of unique answers
    """
    df = pd.read_csv(data_path).dropna(subset=["question", "answer"])
    df["question_clean"] = df["question"].apply(clean_text)
    df["answer_clean"] = df["answer"].apply(clean_text)
    df = df.drop_duplicates(subset=["question_clean", "answer_clean"]).reset_index(drop=True)
    unique_answers = df["answer_clean"].drop_duplicates().tolist()
    ans2id = {a: i for i, a in enumerate(unique_answers)}
    df["answer_id"] = df["answer_clean"].map(ans2id)
    return df, unique_answers
