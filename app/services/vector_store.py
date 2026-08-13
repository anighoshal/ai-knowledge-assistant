from dataclasses import dataclass
import math

@dataclass
class StoredVector:
    text: str
    embedding: list[float]
    document_id: str
    filename: str
    chunk_index: int


class VectorStore:

    def __init__(self):
        self.vectors: list[StoredVector] = []

    def add(
        self,
        text: str,
        embedding: list[float],
        document_id: str,
        filename: str,
        chunk_index: int,
    ) -> None:

        vector = StoredVector(
            text=text,
            embedding=embedding,
            document_id=document_id,
            filename=filename,
            chunk_index=chunk_index,
        )

        self.vectors.append(vector)

    def count(self) -> int:
        return len(self.vectors)

    def get_all(self) -> list[StoredVector]:
        return self.vectors

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple[StoredVector, float]]:

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = []

        for vector in self.vectors:

            similarity = self._cosine_similarity(
                query_embedding,
                vector.embedding,
            )

            results.append(
                (vector, similarity)
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:top_k]

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Vectors must have the same dimensions."
            )

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )
