import json

import pytest

from app.services.embedding_service import EmbeddingService


class FakeBody:

    def read(self):
        return json.dumps(
            {
                "embedding": [0.1, 0.2, 0.3, 0.4]
            }
        ).encode("utf-8")


class FakeBedrockClient:

    def __init__(self):
        self.last_request = None

    def invoke_model(
        self,
        modelId,
        body,
        contentType,
        accept
    ):
        self.last_request = {
            "modelId": modelId,
            "body": json.loads(body),
            "contentType": contentType,
            "accept": accept
        }

        return {
            "body": FakeBody()
        }


def test_embed_text_returns_embedding():

    service = EmbeddingService()

    fake_client = FakeBedrockClient()

    service.client = fake_client

    vector = service.embed_text(
        "Employees receive annual leave."
    )

    assert vector == [0.1, 0.2, 0.3, 0.4]

    assert fake_client.last_request["modelId"] == (
        "amazon.titan-embed-text-v2:0"
    )

    assert fake_client.last_request["body"] == {
        "inputText": "Employees receive annual leave."
    }

    assert fake_client.last_request["contentType"] == (
        "application/json"
    )

    assert fake_client.last_request["accept"] == (
        "application/json"
    )


def test_embed_text_rejects_empty_text():

    service = EmbeddingService()

    service.client = FakeBedrockClient()

    with pytest.raises(
        ValueError,
        match="Text cannot be empty."
    ):
        service.embed_text("")
