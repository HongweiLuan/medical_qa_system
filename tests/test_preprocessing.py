import pytest
import pandas as pd
from medical_qa_system.preprocessing import clean_text, load_and_prepare

@pytest.fixture
def sample_df(tmp_path):
    df = pd.DataFrame({
        "question": ["What is diabetes?", "How to treat cold?"],
        "answer": ["High blood sugar disease.", "Rest and hydration."]
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"

def test_load_and_prepare(sample_df):
    df, unique_answers = load_and_prepare(sample_df)
    assert "answer_id" in df.columns
    assert len(unique_answers) == 2
