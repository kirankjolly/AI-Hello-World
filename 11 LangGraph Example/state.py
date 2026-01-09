from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AppState(TypedDict):
    """
    State object that flows through the entire LangGraph workflow.

    MAGIC #1: State Persistence
    ===========================
    Every time a node updates this state, LangGraph automatically:
    1. Saves a "checkpoint" (snapshot) to the database
    2. Links it to the previous checkpoint (creating a chain)
    3. Associates it with the thread_id from config

    This means the state is NEVER lost - even after app restart!
    """

    # MAGIC #2: The add_messages Reducer
    # ===================================
    # Annotated[List[BaseMessage], add_messages] is the KEY to conversation memory!
    #
    # Without add_messages:
    #   - Each node update would REPLACE the messages list
    #   - Old messages would be lost
    #
    # With add_messages reducer:
    #   - New messages are APPENDED to existing messages
    #   - Old messages are preserved automatically
    #   - Conversation history grows over time
    #
    # Example flow:
    #   Checkpoint 1: messages = [HumanMessage("hi")]
    #   Checkpoint 2: messages = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("who am i")]
    #   Checkpoint 3: messages = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("who am i"), AIMessage("You are...")]
    messages: Annotated[List[BaseMessage], add_messages]

    # Regular state fields (these get REPLACED on each update, not appended)
    question: str              # Current question being processed
    documents: List[Document]  # Retrieved documents from vector store
    answer: str                # Generated answer
    retries: int              # Number of retrieval retries
    needs_retry: bool         # Whether to retry retrieval
