from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "careerai"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# EMBEDDING MODEL
# =========================================================

def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    Streamlit will later cache this model.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF while preserving page information.

    Returns:
        list of dictionaries:
        [
            {
                "text": "...",
                "page": 1
            },
            ...
        ]
    """

    pages = []

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            page_text = page_text.strip()

            if page_text:
                pages.append(
                    {
                        "text": page_text,
                        "page": page_number
                    }
                )

    return pages


def extract_text_from_docx(file_path):
    """
    Extract text from a DOCX file.

    DOCX paragraph-level page numbers are not reliably available
    without a more complex document parser, so page is stored as None.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    if not paragraphs:
        return []

    return [
        {
            "text": "\n".join(paragraphs),
            "page": None
        }
    ]


def extract_text_from_txt(file_path):
    """
    Extract text from a TXT file.
    """

    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None
        }
    ]


# =========================================================
# GENERIC FILE EXTRACTION
# =========================================================

def extract_document(file_path):
    """
    Detect file type and extract its text.
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    if extension == ".docx":
        return extract_text_from_docx(file_path)

    if extension == ".txt":
        return extract_text_from_txt(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported formats are PDF, DOCX and TXT."
    )


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(text, chunk_size=700, overlap=100):
    """
    Split text into overlapping chunks.

    Character-based chunking is intentionally kept simple and
    reliable for the current project.
    """

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    """
    Convert extracted pages into chunks while preserving
    page information.
    """

    chunks = []

    for page_data in extracted_pages:

        text = page_data["text"]
        page_number = page_data["page"]

        page_chunks = chunk_text(text)

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "page": page_number
                }
            )

    return chunks


# =========================================================
# CHROMADB
# =========================================================

def get_chroma_client():
    """
    Return the persistent ChromaDB client.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


def get_collection(client=None):
    """
    Safely get or create the CareerAI collection.

    This prevents:
        NotFoundError: Collection [careerai] does not exist
    """

    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "CareerAI resume knowledge base"
        }
    )

    return collection


# =========================================================
# FILE HASH
# =========================================================

def calculate_file_hash(file_path):
    """
    Generate a SHA-256 hash for a file.

    This helps identify whether a resume has changed.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for block in iter(
            lambda: file.read(8192),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# =========================================================
# INDEX A RESUME
# =========================================================

def index_resume_file(
    file_path,
    embedding_model=None,
    replace_existing=True
):
    """
    Extract, chunk, embed and store a resume in ChromaDB.

    This function will later be used by the Streamlit
    resume-upload feature.

    Parameters:
        file_path:
            Path to PDF/DOCX/TXT.

        embedding_model:
            Existing SentenceTransformer model.
            If None, it will be loaded.

        replace_existing:
            If True, the current knowledge base is replaced
            by the newly uploaded resume.

    Returns:
        embedding_model, collection, number_of_chunks
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if file_path.suffix.lower() not in {
        ".pdf",
        ".docx",
        ".txt"
    }:
        raise ValueError(
            "Unsupported file type. "
            "Please use PDF, DOCX or TXT."
        )

    print(f"Loading resume: {file_path.name}")

    # -----------------------------------------------------
    # Extract
    # -----------------------------------------------------

    extracted_pages = extract_document(file_path)

    if not extracted_pages:
        raise ValueError(
            "The uploaded resume contains no readable text."
        )

    total_characters = sum(
        len(page["text"])
        for page in extracted_pages
    )

    print(
        f"Extracted characters: {total_characters}"
    )

    # -----------------------------------------------------
    # Chunk
    # -----------------------------------------------------

    chunks = create_chunks(extracted_pages)

    if not chunks:
        raise ValueError(
            "No usable text chunks could be created "
            "from the resume."
        )

    print(
        f"Created chunks: {len(chunks)}"
    )

    # -----------------------------------------------------
    # Embedding model
    # -----------------------------------------------------

    if embedding_model is None:
        embedding_model = load_embedding_model()

    texts = [
        item["text"]
        for item in chunks
    ]

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # -----------------------------------------------------
    # ChromaDB
    # -----------------------------------------------------

    chroma_client = get_chroma_client()

    collection = get_collection(
        chroma_client
    )

    # -----------------------------------------------------
    # Replace existing knowledge base
    # -----------------------------------------------------

    if replace_existing:

        existing_data = collection.get()

        existing_ids = existing_data.get(
            "ids",
            []
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )

            print(
                f"Removed {len(existing_ids)} old chunks."
            )

    # -----------------------------------------------------
    # Generate IDs
    # -----------------------------------------------------

    file_hash = calculate_file_hash(
        file_path
    )[:12]

    ids = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{file_hash}_chunk_{index}"
        )

        ids.append(chunk_id)

        metadatas.append(
            {
                "source": file_path.name,
                "file_hash": file_hash,
                "chunk": index,
                "page": chunk["page"]
            }
        )

    # -----------------------------------------------------
    # Store in ChromaDB
    # -----------------------------------------------------

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        "ChromaDB collection updated successfully!"
    )

    print(
        f"Collection name: {COLLECTION_NAME}"
    )

    print(
        f"Documents stored: {collection.count()}"
    )

    return (
        embedding_model,
        collection,
        len(chunks)
    )


# =========================================================
# BUILD INITIAL DATABASE
# =========================================================

def build_database():
    """
    Build the initial database from the resume
    located directly inside data/.

    IMPORTANT:
    If a collection already contains documents,
    it will NOT rebuild it automatically.
    """

    supported_files = (
        list(DATA_DIR.glob("*.pdf"))
        + list(DATA_DIR.glob("*.PDF"))
        + list(DATA_DIR.glob("*.docx"))
        + list(DATA_DIR.glob("*.DOCX"))
        + list(DATA_DIR.glob("*.txt"))
        + list(DATA_DIR.glob("*.TXT"))
    )

    if not supported_files:

        raise FileNotFoundError(
            "No resume found inside the data folder.\n\n"
            "Put your resume PDF, DOCX or TXT file inside:\n"
            f"{DATA_DIR}"
        )

    # Use the first supported resume.
    resume_file = supported_files[0]

    print(
        f"Resume found: {resume_file.name}"
    )

    chroma_client = get_chroma_client()

    collection = get_collection(
        chroma_client
    )

    # -----------------------------------------------------
    # Don't accidentally duplicate existing data
    # -----------------------------------------------------

    existing_count = collection.count()

    if existing_count > 0:

        print(
            f"Collection already contains "
            f"{existing_count} documents."
        )

        print(
            "Existing database will be preserved."
        )

        embedding_model = load_embedding_model()

        return (
            embedding_model,
            collection
        )

    # -----------------------------------------------------
    # Build database
    # -----------------------------------------------------

    embedding_model = load_embedding_model()

    embedding_model, collection, _ = index_resume_file(
        file_path=resume_file,
        embedding_model=embedding_model,
        replace_existing=False
    )

    return (
        embedding_model,
        collection
    )


# =========================================================
# LOAD EXISTING DATABASE
# =========================================================

def load_database():
    """
    Safely load the existing CareerAI database.

    If the collection does not exist, it will be created.
    """

    embedding_model = load_embedding_model()

    chroma_client = get_chroma_client()

    collection = get_collection(
        chroma_client
    )

    return (
        embedding_model,
        collection
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CareerAI Database Builder")
    print("=" * 60)

    try:

        embedding_model, collection = build_database()

        print()
        print("=" * 60)
        print("DATABASE READY")
        print("=" * 60)
        print(
            f"Collection: {COLLECTION_NAME}"
        )
        print(
            f"Documents: {collection.count()}"
        )
        print(
            f"Database path: {CHROMA_DIR}"
        )
        print("=" * 60)

    except Exception as error:

        print()
        print("=" * 60)
        print("DATABASE BUILD FAILED")
        print("=" * 60)
        print(error)
        print("=" * 60)

        raise