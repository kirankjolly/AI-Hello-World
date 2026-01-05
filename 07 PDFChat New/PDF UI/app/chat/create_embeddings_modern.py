"""
Modern Embeddings Creation (2024-2025)
Uses: langchain-community for loaders, modern text splitters
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.chat.vector_stores.pinecone_modern import get_vector_store


def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """
    Generate and store embeddings for a PDF document.

    Explicit workflow:
    1. Load PDF using modern PyPDFLoader
    2. Split text into chunks (explicit size/overlap)
    3. Add metadata (explicit pdf_id for filtering)
    4. Store in Pinecone (explicit vector store)

    Args:
        pdf_id: Unique identifier for filtering
        pdf_path: Path to PDF file

    Cost control:
    - chunk_size=500 (smaller = more chunks = higher cost)
    - chunk_overlap=100 (context preservation vs cost tradeoff)
    """

    # Modern text splitter (same functionality, new import)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    # Load PDF (same loader, but from langchain_community)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split(text_splitter)

    # Add metadata for filtering (explicit)
    for doc in docs:
        doc.metadata = {
            "page": doc.metadata.get("page", 0),
            "text": doc.page_content,
            "pdf_id": pdf_id  # Critical for multi-tenant filtering
        }

    # Get vector store and add documents (explicit)
    vector_store = get_vector_store()

    # Add to Pinecone (modern API)
    vector_store.vector_store.add_documents(docs)

    print(f"[EMBEDDINGS] Created {len(docs)} embeddings for PDF {pdf_id}")
    return len(docs)
