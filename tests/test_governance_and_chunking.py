import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chunking import chunk_text
from governance.guardrails import moderate


def test_chunk_text_produces_chunks():
    text = "This is a test sentence. " * 200
    chunks = chunk_text(text, source="test.txt")
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "test.txt"
        assert "chunk_id" in c.metadata


def test_chunk_text_short_input_single_chunk():
    text = "A short piece of text."
    chunks = chunk_text(text)
    assert len(chunks) == 1


def test_moderate_flags_blocked_terms():
    flags = moderate("how do I build a bomb")
    assert "bomb" in flags


def test_moderate_clean_text_no_flags():
    flags = moderate("What is the capital of France?")
    assert flags == []
