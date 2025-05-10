"""
Extracts human-readable text from an EPUB while stripping
page numbers, running headers/footers, and other audiobook-irrelevant noise.
Optionally pipes paragraphs through a Llama-3 relevance classifier.
"""
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
import re
from typing import List, Optional

NON_CONTENT_PAT = re.compile(
    r"^\s*(page\s*\d+|\d+\s*/\s*\d+|copyright|\u00a9|\*{3,}|table of contents)\s*$",
    re.IGNORECASE,
)

def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    # Further clean the extracted text
    text = re.sub(r'https?://\S+|www\.\S+', ' link ', text) # Replace URLs
    text = re.sub(r'©', ' copyright ', text) # Replace copyright symbol
    text = re.sub(r'[•·]', ' ', text) # Replace bullets
    text = re.sub(r'\d{3}-\d{1}-\d{4}-\d{4}-\d{1}', ' isbn ', text) # Basic ISBN-13 pattern
    text = re.sub(r'\d{1}-\d{5}-\d{3}-\d{1}', ' isbn ', text) # Basic ISBN-10 pattern
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

    return text

def load_epub(path: str) -> List[str]:
    book = epub.read_epub(path)
    chunks = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        text = _html_to_text(item.get_content().decode())
        for line in text.splitlines():
            if line and not NON_CONTENT_PAT.match(line):
                chunks.append(line)
    return chunks

# ---------- OPTIONAL Llama-3 filtering ----------
def llama_filter(chunks: List[str], pipeline=None) -> List[str]:
    """
    Ask Llama-3 whether each paragraph belongs in an audiobook.
    Expect 'yes' / 'no'.  Uses a short prompt for speed.
    """
    if pipeline is None:
        return chunks  # skip if LLM not supplied
    kept = []
    sys_msg = "You clean books for audiobook production."
    for para in chunks:
        prompt = f"{sys_msg}\n\nPARAGRAPH:\n\"\"\"\n{para}\n\"\"\"\nKeep? Answer yes or no."
        answer = pipeline(prompt, max_new_tokens=1, do_sample=False)[0]["generated_text"]
        if answer.strip().lower().startswith("yes"):
            kept.append(para)
    return kept
