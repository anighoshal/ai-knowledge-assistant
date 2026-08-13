from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_text_file(file_path: str) -> str:
    """
    Read a plain text file and return its contents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


def load_pdf_file(file_path: str) -> str:
    """
    Extract text from all pages of a PDF file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_document(file_path: str) -> str:
    """
    Load a supported document and return its extracted text.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_text_file(file_path)

    if extension == ".pdf":
        return load_pdf_file(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
