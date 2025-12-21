from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query: str) -> List[Document]:
        emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            fetch_k=20,
            k=6,
            lambda_mult=0.8
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
