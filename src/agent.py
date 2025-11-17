"""
RAG Agent with STM, LTM, and Entity Memory.
"""

from typing import List, Tuple, Dict, Any

import numpy as np

from .vector_store import FaissVectorStore, DocumentChunk
from .embeddings import EmbeddingModel
from .memory import STMMemory, LTMMemory, EntityMemory


class SimpleHeuristicGenerator:
    """
    Very simple "generator":
    - ranks sentences from retrieved chunks by similarity to the question
      using an evaluation embedding model (passed separately).
    - returns the best-matching sentence plus some context.
    """

    def __init__(self, eval_model):
        self.eval_model = eval_model

    def generate(
        self,
        question: str,
        retrieved_chunks: List[DocumentChunk],
        stm_context: str,
        ltm_snippets: List[str],
        entity_summary: str,
    ) -> str:
        if not retrieved_chunks:
            return (
                "I cannot find relevant information in the thesis proposal for this question."
            )

        sentences: List[str] = []
        for ch in retrieved_chunks[:3]:
            for sent in ch.text.split("."):
                s = sent.strip()
                if len(s) < 20:
                    continue
                sentences.append(s)

        if not sentences:
            best_excerpt = retrieved_chunks[0].text[:400]
        else:
            q_emb = self.eval_model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            sent_embs = self.eval_model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sims = np.dot(sent_embs, q_emb)
            best_idx = int(np.argmax(sims))
            best_excerpt = sentences[best_idx]

        parts = [
            "Based on the thesis proposal, here is a concise answer:",
            "",
            best_excerpt.strip() + ".",
        ]

        if ltm_snippets:
            parts.append("")
            parts.append("Additional relevant details remembered from prior QA:")
            parts.extend(ltm_snippets)

        if entity_summary:
            parts.append("")
            parts.append("Key entities mentioned so far:")
            parts.append(entity_summary)

        return "\n".join(parts)


class RAGAgent:
    """
    Stateful RAG agent with:
    - vector store retriever,
    - STM,
    - optional LTM,
    - entity memory,
    - heuristic generator.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embed_model: EmbeddingModel,
        stm: STMMemory,
        ltm: LTMMemory,
        entities: EntityMemory,
        generator: SimpleHeuristicGenerator,
        top_k: int = 5,
        use_ltm: bool = True,
    ):
        self.vs = vector_store
        self.embed_model = embed_model
        self.stm = stm
        self.ltm = ltm
        self.entities = entities
        self.generator = generator
        self.top_k = top_k
        self.use_ltm = use_ltm

    def _retrieve(self, question: str) -> Tuple[List[DocumentChunk], List[float]]:
        q_emb = self.embed_model.encode(question)
        results = self.vs.search(q_emb, top_k=self.top_k)
        chunks = [c for c, _ in results]
        scores = [s for _c, s in results]
        return chunks, scores

    def answer(self, question: str, user_id: str = "default_user") -> Tuple[str, Dict[str, Any]]:
        # 1. STM: add question
        self.stm.add_message("user", question)

        # 2. LTM retrieval
        ltm_snippets: List[str] = []
        if self.use_ltm and self.ltm is not None:
            items = self.ltm.retrieve(question, top_k=3)
            ltm_snippets = [it.text for it in items]

        # 3. Retrieve thesis chunks
        chunks, scores = self._retrieve(question)
        stm_context = self.stm.get_context_text()
        entity_summary = self.entities.get_entity_summary()

        # 4. Generate answer
        answer = self.generator.generate(
            question=question,
            retrieved_chunks=chunks,
            stm_context=stm_context,
            ltm_snippets=ltm_snippets,
            entity_summary=entity_summary,
        )

        # 5. STM: add answer
        self.stm.add_message("assistant", answer)

        # 6. LTM write policy
        avg_score = float(np.mean(scores)) if scores else 0.0
        if self.use_ltm and self.ltm is not None:
            self.ltm.write_policy_maybe_save(
                question=question,
                answer=answer,
                retrieval_confidence=avg_score,
                tags=["thesis_qa"],
            )

        # 7. Entity usage tracking (very simple)
        for name in self.entities.entities.keys():
            if name.lower() in answer.lower():
                self.entities.record_mention(name)

        debug = {
            "retrieved_chunks": [c.id for c in chunks],
            "retrieval_scores": scores,
            "stm_context": stm_context,
            "ltm_snippets": ltm_snippets,
        }
        return answer, debug
