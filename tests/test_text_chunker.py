import pytest

from app.services.text_chunker import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = "A" * 2500

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) == 4


def test_chunks_have_correct_indexes():
    text = "A" * 2500

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200,
    )

    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2


def test_empty_text_returns_empty_list():
    chunks = chunk_text("")

    assert chunks == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text",
            chunk_size=100,
            overlap=100,
        )
