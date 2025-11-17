"""
Evaluation utilities: mapping gold answers to chunks, running ablations.
"""

from dataclasses import dataclass
from typing import List, Dict

import time
import numpy as np
import pandas as pd

from .vector_store import DocumentChunk
from .agent import RAGAgent
from .data_model import QuestionItem


def build_gold_chunk_mapping(
    questions: List[QuestionItem],
    docs: List[DocumentChunk],
    eval_embed_model,
) -> Dict[int, str]:
    """
    Approximate mapping from question ID -> gold-supporting chunk ID
    by matching gold answers to document chunks in embedding space.
    """
    gold_answers = [q.gold_answer for q in questions]
    ga_embs = eval_embed_model.encode(
        gold_answers,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    doc_texts = [d.text for d in docs]
    doc_embs = eval_embed_model.encode(
        doc_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    mapping: Dict[int, str] = {}

    for i, q in enumerate(questions):
        sims = np.dot(doc_embs, ga_embs[i])
        best_idx = int(np.argmax(sims))
        mapping[q.id] = docs[best_idx].id

    return mapping


def evaluate_agent_on_questions(
    agent: RAGAgent,
    questions: List[QuestionItem],
    gold_chunk_mapping: Dict[int, str],
    eval_embed_model,
    top_k: int,
    experiment_name: str,
) -> pd.DataFrame:
    """
    Evaluate an agent on a question set.

    Returns DataFrame with columns:
    - experiment_name
    - question_id
    - hit_at_k
    - reciprocal_rank
    - answer_cosine_sim
    - latency
    """
    rows = []

    for q in questions:
        t0 = time.time()
        answer, debug = agent.answer(q.question)
        latency = time.time() - t0

        retrieved_ids = debug["retrieved_chunks"]
        gold_id = gold_chunk_mapping[q.id]

        # Hit@k
        hit_at_k = 1.0 if gold_id in retrieved_ids[:top_k] else 0.0

        # Reciprocal Rank
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid == gold_id:
                rr = 1.0 / rank
                break

        # Answer similarity
        ans_emb = eval_embed_model.encode(
            [answer],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        gold_emb = eval_embed_model.encode(
            [q.gold_answer],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        ans_sim = float(np.dot(ans_emb, gold_emb))

        rows.append(
            {
                "experiment_name": experiment_name,
                "question_id": q.id,
                "hit_at_k": hit_at_k,
                "reciprocal_rank": rr,
                "answer_cosine_sim": ans_sim,
                "latency": latency,
            }
        )

    return pd.DataFrame(rows)


def aggregate_results(all_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-experiment metrics.
    """
    agg = all_results_df.groupby("experiment_name").agg(
        hit_rate_at_k=("hit_at_k", "mean"),
        mrr=("reciprocal_rank", "mean"),
        answer_sim_mean=("answer_cosine_sim", "mean"),
        latency_p50=("latency", lambda x: np.percentile(x, 50)),
        latency_p95=("latency", lambda x: np.percentile(x, 95)),
    )
    return agg.reset_index()
