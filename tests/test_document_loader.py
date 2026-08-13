from app.services.document_loader import load_document

def test_load_text_document():
  text = load_document("data/documents/sample.txt")

  assert "AI Knowledge Assistant" in text
  assert len(text) > 0
