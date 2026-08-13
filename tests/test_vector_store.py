from app.services.vector_store import VectorStore


def test_vector_store_starts_empty():
    store = VectorStore()

    assert store.count() == 0


def test_vector_store_can_add_vector():
    store = VectorStore()

    store.add(
        text="Employees receive annual leave.",
        embedding=[0.1, 0.2, 0.3],
        document_id="doc-001",
        filename="employee_policy.pdf",
        chunk_index=0,
    )

    assert store.count() == 1


def test_vector_store_stores_metadata():
    store = VectorStore()

    store.add(
        text="Employees receive annual leave.",
        embedding=[0.1, 0.2, 0.3],
        document_id="doc-001",
        filename="employee_policy.pdf",
        chunk_index=0,
    )

    vectors = store.get_all()

    assert vectors[0].text == "Employees receive annual leave."
    assert vectors[0].embedding == [0.1, 0.2, 0.3]
    assert vectors[0].document_id == "doc-001"
    assert vectors[0].filename == "employee_policy.pdf"
    assert vectors[0].chunk_index == 0


def test_vector_store_returns_most_similar_vector_first():
    store = VectorStore()

    store.add(
        text="Similar document",
        embedding=[1.0, 0.0, 0.0],
        document_id="doc-001",
        filename="similar.txt",
        chunk_index=0,
    )

    store.add(
        text="Different document",
        embedding=[0.0, 1.0, 0.0],
        document_id="doc-002",
        filename="different.txt",
        chunk_index=0,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert results[0][0].text == "Similar document"
    assert results[0][1] > results[1][1]


def test_vector_store_respects_top_k():
    store = VectorStore()

    store.add(
        text="Document 1",
        embedding=[1.0, 0.0, 0.0],
        document_id="doc-001",
        filename="one.txt",
        chunk_index=0,
    )

    store.add(
        text="Document 2",
        embedding=[0.9, 0.1, 0.0],
        document_id="doc-002",
        filename="two.txt",
        chunk_index=0,
    )

    store.add(
        text="Document 3",
        embedding=[0.8, 0.2, 0.0],
        document_id="doc-003",
        filename="three.txt",
        chunk_index=0,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_search_rejects_empty_query_embedding():
    store = VectorStore()

    try:
        store.search([])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query embedding cannot be empty."


def test_vector_store_ranks_multiple_vectors_correctly():
    store = VectorStore()

    store.add(
        text="Very similar document",
        embedding=[0.95, 0.05, 0.0],
        document_id="doc-001",
        filename="similar.txt",
        chunk_index=0,
    )

    store.add(
        text="Somewhat similar document",
        embedding=[0.7, 0.3, 0.0],
        document_id="doc-002",
        filename="somewhat_similar.txt",
        chunk_index=0,
    )

    store.add(
        text="Unrelated document",
        embedding=[0.0, 0.0, 1.0],
        document_id="doc-003",
        filename="unrelated.txt",
        chunk_index=0,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert results[0][0].text == "Very similar document"
    assert results[1][0].text == "Somewhat similar document"
    assert results[2][0].text == "Unrelated document"

    assert results[0][1] > results[1][1]
    assert results[1][1] > results[2][1]
