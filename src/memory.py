"""
Memory components: STM, LTM, and entity memory.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import json
import os
import numpy as np

from .vector_store import FaissVectorStore, DocumentChunk
from .embeddings import EmbeddingModel


def approximate_token_count(text: str) -> int:
    return len(text.split())


class STMMemory:
    """
    Short-term memory: rolling buffer constrained by a token budget (approx).
    """

    def __init__(self, max_tokens: int = 800):
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        while True:
            total = sum(approximate_token_count(m["content"]) for m in self.messages)
            if total <= self.max_tokens or not self.messages:
                break
            self.messages.pop(0)

    def get_context_text(self) -> str:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in self.messages]
        return "\n".join(lines)


@dataclass
class LTMMemoryItem:
    id: str
    text: str
    tags: List[str]


class LTMMemory:
    """
    Long-term memory: simple JSON-backed store + optional vector store
    for semantic retrieval of remembered items.
    """

    def __init__(self, json_path: str, embedding_model: Optional[EmbeddingModel] = None):
        self.json_path = json_path
        self.embedding_model = embedding_model
        self.items: Dict[str, LTMMemoryItem] = {}
        self.vector_store: Optional[FaissVectorStore] = None

        self._load()

    def _load(self) -> None:
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                item = LTMMemoryItem(**d)
                self.items[item.id] = item

        if self.embedding_model is not None and self.items:
            dim = self.embedding_model.dimension
            self.vector_store = FaissVectorStore(dimension=dim)
            docs = [
                DocumentChunk(id=i.id, text=i.text, metadata={"tags": i.tags})
                for i in self.items.values()
            ]
            self.vector_store.build_index(
                docs=docs,
                embed_fn=self.embedding_model.encode_texts,
            )

    def _save(self) -> None:
        data = [asdict(i) for i in self.items.values()]
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def upsert_item(self, item_id: str, text: str, tags: Optional[List[str]] = None):
        tags = tags or []
        self.items[item_id] = LTMMemoryItem(id=item_id, text=text, tags=tags)
        self._save()

        if self.embedding_model is not None:
            if self.vector_store is None:
                self.vector_store = FaissVectorStore(dimension=self.embedding_model.dimension)
            doc = DocumentChunk(id=item_id, text=text, metadata={"tags": tags})
            self.vector_store.add_docs([doc], embed_fn=self.embedding_model.encode_texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[LTMMemoryItem]:
        if self.vector_store is None or self.embedding_model is None or not self.items:
            return list(self.items.values())[:top_k]

        q_emb = self.embedding_model.encode(query)
        results = self.vector_store.search(q_emb, top_k=top_k)
        out: List[LTMMemoryItem] = []
        for doc, _score in results:
            item = self.items.get(doc.id)
            if item:
                out.append(item)
        return out

    def write_policy_maybe_save(
        self,
        question: str,
        answer: str,
        retrieval_confidence: float,
        tags: Optional[List[str]] = None,
    ) -> None:
        tags = tags or []
        # Simple heuristic: only save if retrieval was reasonably confident
        if retrieval_confidence < 0.4:
            return

        core_keywords = [
            "differential privacy",
            "rag",
            "membership inference",
            "dp-mlm",
            "dp-bart",
            "enron",
            "healthcaremagic",
        ]
        if not any(kw in question.lower() for kw in core_keywords):
            return

        item_id = f"ltm_{len(self.items)}"
        text = f"Q: {question}\nA: {answer}"
        self.upsert_item(item_id=item_id, text=text, tags=tags)


class EntityMemory:
    """
    Very lightweight entity store / KG-lite.

    {
      "entity_name": {
          "type": str,
          "attributes": {...},
          "mentions": int
      }
    }
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        attributes = attributes or {}
        if name not in self.entities:
            self.entities[name] = {
                "type": entity_type,
                "attributes": attributes,
                "mentions": 0,
            }
        else:
            self.entities[name]["type"] = entity_type
            self.entities[name]["attributes"].update(attributes)

    def record_mention(self, name: str) -> None:
        if name in self.entities:
            self.entities[name]["mentions"] += 1

    def get_entity_summary(self) -> str:
        lines = []
        for name, info in self.entities.items():
            ent_type = info.get("type", "entity")
            attrs = info.get("attributes", {})
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
            lines.append(f"{name} ({ent_type}): {attr_str}")
        return "\n".join(lines)
