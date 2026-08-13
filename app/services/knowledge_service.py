from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class KnowledgeService:

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

    def add_document_chunk(
        self,
        text: str,
        document_id: str,
        filename: str,
        chunk_index: int,
    ) -> None:

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.embedding_service.embed_text(text)

        self.vector_store.add(
            text=text,
            embedding=embedding,
            document_id=document_id,
            filename=filename,
            chunk_index=chunk_index,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_service.embed_text(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
