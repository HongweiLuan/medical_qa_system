import pytest
from medical_qa_system.training import build_training_examples
import pandas as pd

@pytest.fixture
def df():
    return pd.DataFrame({
        "question_clean": ["q1", "q2"],
        "answer_clean": ["a1", "a2"]
    })

def test_build_training_examples(df):
    examples = build_training_examples(df)
    assert len(examples) == 2
    assert all(hasattr(e, "texts") for e in examples)
