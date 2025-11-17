"""
Chunking strategies for the thesis corpus.

We implement:
- Fixed-size character chunks with overlap.
- Structure-aware chunking based on headings like "Abstract", "Chapter 1", etc.
"""

import re
from typing import List, Dict, Any


class FixedSizeChunker:
    """
    Simple fixed-size chunker by characters with overlap.
    """

    def __init__(self, chunk_size_chars: int = 1000, overlap_chars: int = 200):
        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        start = 0
        n = len(text)
        idx = 0

        while start < n:
            end = min(start + self.chunk_size_chars, n)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "id": f"fixed_{idx}",
                        "text": chunk_text,
                        "metadata": {
                            "strategy": "fixed",
                            "section": f"fixed_{idx}",
                        },
                    }
                )
                idx += 1

            if end == n:
                break
            start = end - self.overlap_chars

        return chunks


class StructureAwareChunker:
    """
    Heading-aware chunker specialized to thesis-like documents.

    It uses regex to detect:
    - Abstract
    - Acknowledgments
    - Chapter headings ("Chapter 1", "Chapter 2", ...)
    - Numbered sections like "1 Introduction" or "1.1 Background"

    Then merges adjacent segments to keep chunk sizes within [min_chunk_chars, max_chunk_chars].
    """

    SECTION_HEADER_RE = re.compile(
        r"(?m)^(Abstract|Acknowledgments|Chapter\s+\d+|[0-9]+(\.[0-9]+)*\s+.+)$"
    )

    def __init__(self, min_chunk_chars: int = 400, max_chunk_chars: int = 1600):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        matches = list(self.SECTION_HEADER_RE.finditer(text))
        if not matches:
            return [
                {
                    "id": "struct_0",
                    "text": text.strip(),
                    "metadata": {"strategy": "structure", "section": "full_document"},
                }
            ]

        segments = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            header = m.group(0).strip()
            seg_text = text[start:end].strip()
            if seg_text:
                segments.append((header, seg_text))

        chunks: List[Dict[str, Any]] = []
        buffer_text = ""
        buffer_header = None
        idx = 0

        for header, seg_text in segments:
            if not buffer_text:
                buffer_header = header
                buffer_text = seg_text
            else:
                if len(buffer_text) + len(seg_text) <= self.max_chunk_chars:
                    buffer_text += "\n\n" + seg_text
                else:
                    if len(buffer_text) < self.min_chunk_chars and chunks:
                        chunks[-1]["text"] += "\n\n" + buffer_text
                    else:
                        chunks.append(
                            {
                                "id": f"struct_{idx}",
                                "text": buffer_text,
                                "metadata": {
                                    "strategy": "structure",
                                    "section": buffer_header,
                                },
                            }
                        )
                        idx += 1
                    buffer_header = header
                    buffer_text = seg_text

        if buffer_text:
            chunks.append(
                {
                    "id": f"struct_{idx}",
                    "text": buffer_text,
                    "metadata": {
                        "strategy": "structure",
                        "section": buffer_header,
                    },
                }
            )

        return chunks
