from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    document_id: str
    filename: str


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    document_id: str = "unknown",
    filename: str = "unknown",
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of TextChunk objects.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    text=chunk,
                    chunk_index=chunk_index,
                    document_id = document_id,
                    filename = filename,
                )
            )

            chunk_index += 1

        start += step

    return chunks
