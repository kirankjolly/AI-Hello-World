from graph import build_graph
from ingest import ingest_pdfs
from langchain_core.messages import HumanMessage

# Run PDF ingestion on startup
print("Checking for new PDFs to ingest...")
ingest_pdfs()
print("Building graph...")

# Build the graph ONCE at startup
# This creates the compiled graph with checkpointer attached
app = build_graph()

def ask(question: str, thread_id: str = "default"):
    """
    Ask a question with conversation memory.

    Args:
        question: The user's question
        thread_id: Conversation thread ID (use same ID to continue a conversation)

    MAGIC #7: The thread_id - Conversation Identity
    ================================================
    The thread_id is the KEY to conversation memory!

    How it works:
    1. You pass thread_id in the config parameter
    2. LangGraph's checkpointer uses it to:
       - Look up the latest checkpoint for this thread_id in the database
       - Load the previous state (including all messages)
       - Merge new input with old state
       - Save updated state back with the same thread_id

    Example:
    --------
    # First call with thread_id="session_1"
    ask("hi", thread_id="session_1")
    → Database: session_1 → [HumanMessage("hi"), AIMessage("hello")]

    # Second call with SAME thread_id
    ask("who am i", thread_id="session_1")
    → LangGraph loads: [HumanMessage("hi"), AIMessage("hello")]
    → Adds new message: [HumanMessage("hi"), AIMessage("hello"), HumanMessage("who am i")]
    → Saves back to database

    # Call with DIFFERENT thread_id
    ask("hi", thread_id="session_2")
    → Fresh start! No history from session_1
    """

    # MAGIC #8: The config Parameter
    # ===============================
    # config = {"configurable": {"thread_id": thread_id}}
    #
    # This tells LangGraph:
    # - Which conversation this belongs to
    # - Where to load previous state from
    # - Where to save new state to
    #
    # The format {"configurable": {"thread_id": ...}} is required by LangGraph
    config = {"configurable": {"thread_id": thread_id}}

    # MAGIC #9: State Merging
    # ========================
    # We pass a new state dict here:
    # {
    #   "messages": [HumanMessage(content=question)],  ← New message
    #   "question": question,
    #   "retries": 0
    # }
    #
    # LangGraph will:
    # 1. Load old state from database (if thread_id exists)
    # 2. Merge new state with old state:
    #    - For "messages": Uses add_messages reducer → APPENDS new message
    #    - For "question", "retries": REPLACES old value with new value
    # 3. Execute the graph with the merged state
    # 4. Save final state back to database
    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "retries": 0
    }, config)

    # Return just the answer to the user
    # But the full state (including all messages) is saved in the database!
    return result["answer"]
