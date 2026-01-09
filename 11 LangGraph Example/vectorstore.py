from langchain_community.vectorstores import Chroma
from embeddings import get_embeddings
from config import CHROMA_PATH

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )
