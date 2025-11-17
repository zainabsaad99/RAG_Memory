"""
Data ingestion utilities.

This module loads the thesis proposal PDF and extracts its text.
We use pypdf for simplicity and robustness.

The resulting text is a single string which will later be chunked.
"""

from typing import List
from pypdf import PdfReader


def load_pdf_text(path: str) -> str:
    """
    Load and concatenate text from a PDF file.

    Parameters
    ----------
    path : str
        Path to the PDF file on disk.

    Returns
    -------
    str
        Full concatenated text of all pages.
    """
    reader = PdfReader(path)
    pages_text: List[str] = []
    for page in reader.pages:
        # `extract_text()` is robust and fine for our academic text
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n\n".join(pages_text)
