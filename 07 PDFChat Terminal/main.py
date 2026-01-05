import os
import logging
import hashlib
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------- SETUP ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"

# ---------------- UTILS ----------------
def file_hash(path: str) -> str:
    """Create a stable hash of the PDF file"""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def pdf_already_indexed(db, pdf_hash: str) -> bool:
    """Check if this PDF was already embedded"""
    results = db.get(where={"pdf_hash": pdf_hash})
    return len(results["ids"]) > 0

# ---------------- INIT LLM ----------------
def init_llm():
    logger.info("Initializing OpenAI LLM...")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=256
    )

# ---------------- INIT EMBEDDINGS ----------------
def init_embeddings():
    logger.info("Initializing OpenAI embeddings...")
    return OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

# ---------------- BUILD / LOAD VECTOR STORE ----------------
def build_vector_store(pdf_path: str, embeddings):
    pdf_hash = file_hash(pdf_path)

    # Always load or create Chroma
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Deduplication check
    if os.path.exists(CHROMA_DIR) and pdf_already_indexed(db, pdf_hash):
        logger.info("PDF already indexed. Reusing existing embeddings.")
        return db

    logger.info("Indexing new PDF: %s", pdf_path)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )
    chunks = splitter.split_documents(docs)

    # Attach hash metadata for dedupe
    for chunk in chunks:
        chunk.metadata["pdf_hash"] = pdf_hash
        chunk.metadata["source"] = pdf_path

    db.add_documents(chunks)
    db.persist()

    logger.info("PDF indexed and persisted successfully.")
    return db

# ---------------- RETRIEVE ----------------
def retrieve_context(db, question: str, k: int = 6):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.25}
    )
    return retriever.invoke(question)

# ---------------- GENERATE ANSWER ----------------
def generate_answer(llm, question: str, documents):
    context = "\n\n".join(doc.page_content for doc in documents)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant.\n"
                "Answer ONLY using the provided context.\n"
                "If the answer is not in the context, say so."
            )
        ),
        SystemMessage(content=f"Context:\n{context}"),
        HumanMessage(content=question),
    ]

    return llm.invoke(messages).content

# ---------------- MAIN WORKFLOW ----------------
def ask_question(llm, db, question: str):
    logger.info("Question: %s", question)
    docs = retrieve_context(db, question)
    logger.info("Retrieved %d documents", len(docs))
    return generate_answer(llm, question, docs)

# ---------------- RUN ----------------
if __name__ == "__main__":
    llm = init_llm()
    embeddings = init_embeddings()
    vector_db = build_vector_store("document.pdf", embeddings)

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        response = ask_question(llm, vector_db, q)
        print("\nAnswer:\n", response)
