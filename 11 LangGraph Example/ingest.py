import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embeddings import get_embeddings
from config import CHROMA_PATH


def get_processed_files(vectorstore):
    """Get list of already processed PDF files from vectorstore metadata."""
    try:
        collection = vectorstore._collection
        all_docs = collection.get(include=['metadatas'])
        if all_docs and all_docs['metadatas']:
            processed = set(
                meta.get('source_file', '')
                for meta in all_docs['metadatas']
                if meta.get('source_file')
            )
            return processed
    except Exception as e:
        print(f"Error getting processed files: {e}")
    return set()


def ingest_pdfs():
    """Load PDFs from data/pdf/ and add new ones to vectorstore."""
    pdf_dir = Path("data/pdf")

    if not pdf_dir.exists():
        print(f"PDF directory {pdf_dir} does not exist")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/pdf/")
        return

    # Initialize vectorstore
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Get already processed files
    processed_files = get_processed_files(vectorstore)

    # Filter out already processed files
    new_files = [f for f in pdf_files if f.name not in processed_files]
    skipped_files = [f for f in pdf_files if f.name in processed_files]

    # Print skipped files
    if skipped_files:
        print(f"\nSkipping {len(skipped_files)} already ingested file(s):")
        for pdf_file in skipped_files:
            print(f"  - {pdf_file.name}")

    if not new_files:
        print(f"\nAll {len(pdf_files)} PDF file(s) already processed. No new files to ingest.")
        return

    print(f"\nFound {len(new_files)} new PDF file(s) to process:")

    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Process each new PDF
    for pdf_file in new_files:
        print(f"Processing: {pdf_file.name}")

        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # Split into chunks
            chunks = text_splitter.split_documents(documents)

            # Add source_file metadata to track processed files
            for chunk in chunks:
                chunk.metadata['source_file'] = pdf_file.name

            # Add to vectorstore
            vectorstore.add_documents(chunks)

            print(f"  -> Added {len(chunks)} chunks from {pdf_file.name}")

        except Exception as e:
            print(f"  -> Error processing {pdf_file.name}: {e}")

    # Explicitly persist to disk (ensures documents are saved)
    print("\nPersisting documents to disk...")
    vectorstore.persist()

    print(f"Ingestion complete. Processed {len(new_files)} new file(s)")


if __name__ == "__main__":
    ingest_pdfs()
