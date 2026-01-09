from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AppState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    answer: str
    retries: int
    needs_retry: bool
