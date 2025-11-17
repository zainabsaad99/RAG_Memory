"""
Global configuration for the Agent Memory + RAG assignment.
"""

import os

# Path to the thesis proposal PDF
DEFAULT_CORPUS_PATH = os.environ.get(
    "THESIS_PDF_PATH",
    os.path.join("data", "proposal_thesis.pdf"),
)

# Embeddings ---------------------------------------------------------

SMALL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LARGE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

VECTOR_DIM_SMALL = 384   # known dim of MiniLM-L6-v2
VECTOR_DIM_LARGE = 768   # known dim of mpnet-base-v2

# For evaluation similarity
EVAL_EMBEDDING_MODEL_NAME = LARGE_EMBEDDING_MODEL_NAME

# Retrieval
EXPERIMENT_TOP_K = 5
