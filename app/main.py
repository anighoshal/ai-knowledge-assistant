from fastapi import FastAPI

app = FastAPI(
    title="AI Knowledge Assistant",
    description="AI-powered document question-answering system",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "AI Knowledge Assistant API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ai-knowledge-assistant"
    }
