import json

import boto3


class EmbeddingService:

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="ap-south-1"
        )

    def embed_text(self, text):

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        response = self.client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text
            }),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(
            response["body"].read()
        )

        return result["embedding"]
