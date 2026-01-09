from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from state import AppState
from nodes.retrieve import retrieve_docs
from nodes.evaluate import evaluate_docs
from nodes.rewrite import rewrite_query
from nodes.generate import generate_answer
from config import MAX_RETRIES
import os

def build_graph():
    """
    Build the LangGraph workflow with conversation memory.

    Graph Flow:
    ===========
    retrieve → evaluate → [rewrite → retrieve (loop)] OR generate → END

    MAGIC #3: Automatic Checkpointing
    =================================
    Because we pass a 'checkpointer' to graph.compile(), LangGraph will:
    1. Save state after EVERY node execution
    2. Associate each save with the thread_id from config
    3. Load previous state automatically when the same thread_id is used

    This is how conversation memory works!
    """

    # Create the state graph
    graph = StateGraph(AppState)

    # Add all the nodes (functions that process state)
    # Each node receives the current state and returns state updates
    graph.add_node("retrieve", retrieve_docs)      # Get relevant docs from vector store
    graph.add_node("evaluate", evaluate_docs)      # Check if docs are good enough
    graph.add_node("rewrite", rewrite_query)       # Improve query if needed
    graph.add_node("generate", generate_answer)    # Generate final answer

    # Define the workflow edges (what happens after each node)
    graph.set_entry_point("retrieve")              # Start here
    graph.add_edge("retrieve", "evaluate")         # Always go to evaluate after retrieve

    # MAGIC #4: Conditional Edges
    # ===========================
    # This decides the next node based on state values
    # If needs_retry=True and retries < MAX: go to "rewrite"
    # Otherwise: go to "generate"
    graph.add_conditional_edges(
        "evaluate",
        lambda s: "rewrite" if s.get("needs_retry") and s.get("retries", 0) < MAX_RETRIES else "generate",
        {"rewrite": "rewrite", "generate": "generate"}
    )

    graph.add_edge("rewrite", "retrieve")          # Loop back to retrieve after rewrite
    graph.add_edge("generate", END)                # End after generating answer

    # Ensure data directory exists for the database
    os.makedirs("data", exist_ok=True)

    # MAGIC #5: SqliteSaver - The Memory System
    # ==========================================
    # SqliteSaver automatically:
    # 1. Creates tables: checkpoints, writes
    # 2. Serializes (pickles) the entire state object to BLOB
    # 3. Saves it with thread_id + checkpoint_id
    # 4. Links each checkpoint to its parent (creating a chain)
    #
    # When you call app.invoke() with a thread_id:
    # - SqliteSaver looks up the latest checkpoint for that thread_id
    # - Deserializes the state from BLOB
    # - Merges it with your new input (using reducers like add_messages)
    # - Continues execution from that state
    #
    # This is why your conversation history persists!
    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # MAGIC #6: Compile with Checkpointer
    # ====================================
    # Passing checkpointer=memory enables:
    # - Automatic state saving after each node
    # - Automatic state loading based on thread_id
    # - Time-travel (you can go back to any checkpoint)
    # - Conversation branching (fork conversations)
    return graph.compile(checkpointer=memory)
