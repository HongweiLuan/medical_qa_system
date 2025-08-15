import pytest
from medical_qa_system.retrieval import build_tfidf_index, build_bm25, retrieve_with_tfidf, retrieve_with_bm25

@pytest.fixture
def answers():
    return ["high blood sugar", "rest and hydration"]

def test_tfidf_retrieval(answers):
    vec, mat = build_tfidf_index(answers)
    res = retrieve_with_tfidf("blood", vec, mat, answers, top_k=1)
    assert "blood" in res[0]

def test_bm25_retrieval(answers):
    bm25 = build_bm25(answers)
    res = retrieve_with_bm25("hydration", bm25, answers, top_k=1)
    assert "hydration" in res[0]
