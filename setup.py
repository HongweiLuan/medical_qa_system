from setuptools import setup, find_packages

setup(
    name="medical_qa_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch>=2.0",
        "scikit-learn",
        "tqdm",
        "rank-bm25",
        "sentence-transformers",
        "matplotlib"
    ],
    python_requires=">=3.8",
    description="Medical QA retrieval system with TF-IDF, BM25, and dense embeddings."
)
