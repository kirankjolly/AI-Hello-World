"""
Modern Pinecone Vector Store (2024-2025)
Uses: Pinecone v5, langchain-pinecone, langchain-openai
"""
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


class ModernPineconeStore:
    """
    Explicit Pinecone vector store wrapper.
    Application controls initialization and retrieval.
    """

    def __init__(self):
        # Modern Pinecone v5 initialization
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )

        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)

        # Modern OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Latest, cheaper model
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Modern LangChain Pinecone integration
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

    def build_retriever(self, pdf_id: str, k: int = 5):
        """
        Build a retriever with explicit filtering and limits.

        Args:
            pdf_id: Filter documents by PDF ID
            k: Number of documents to retrieve (cost control)

        Returns:
            Retriever configured for this PDF
        """
        search_kwargs = {
            "filter": {"pdf_id": pdf_id},
            "k": k
        }

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def get_stats(self):
        """Get index statistics for monitoring."""
        return self.index.describe_index_stats()


# Singleton instance (lazy initialization)
_store_instance = None


def get_vector_store() -> ModernPineconeStore:
    """Get or create vector store singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ModernPineconeStore()
    return _store_instance


def build_retriever(chat_args, k: int = 5):
    """
    Factory function compatible with existing code.
    Explicitly builds retriever with cost controls.
    """
    store = get_vector_store()
    return store.build_retriever(pdf_id=chat_args.pdf_id, k=k)
