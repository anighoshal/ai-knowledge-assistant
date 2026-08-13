from app.services.knowledge_service import KnowledgeService
from app.services.vector_store import VectorStore


class FakeEmbeddingService:

    def __init__(self):
        self.calls = []

    def embed_text(self, text):
        self.calls.append(text)

        if text == "Employees receive annual leave.":
            return [1.0, 0.0, 0.0]

        if text == "What is the annual leave?":
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]


def test_add_document_chunk_creates_embedding_and_stores_vector():
    embedding_service = FakeEmbeddingService()
    vector_store = VectorStore()

    service = KnowledgeService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service.add_document_chunk(
        text="Employees receive annual leave.",
        document_id="doc-001",
        filename="employee_policy.txt",
        chunk_index=0,
    )

    assert embedding_service.calls == [
        "Employees receive annual leave."
    ]

    assert vector_store.count() == 1

    stored = vector_store.get_all()[0]

    assert stored.text == "Employees receive annual leave."
    assert stored.embedding == [1.0, 0.0, 0.0]
    assert stored.document_id == "doc-001"
    assert stored.filename == "employee_policy.txt"
    assert stored.chunk_index == 0


def test_search_embeds_query_and_returns_matching_chunk():
    embedding_service = FakeEmbeddingService()
    vector_store = VectorStore()

    service = KnowledgeService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service.add_document_chunk(
        text="Employees receive annual leave.",
        document_id="doc-001",
        filename="employee_policy.txt",
        chunk_index=0,
    )

    results = service.search(
        "What is the annual leave?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0].text == "Employees receive annual leave."
    assert results[0][1] == 1.0


def test_add_document_chunk_rejects_empty_text():
    service = KnowledgeService(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(),
    )

    try:
        service.add_document_chunk(
            text="",
            document_id="doc-001",
            filename="test.txt",
            chunk_index=0,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Text cannot be empty."


def test_search_rejects_empty_query():
    service = KnowledgeService(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(),
    )

    try:
        service.search("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
