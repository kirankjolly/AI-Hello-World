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
    graph = StateGraph(AppState)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("evaluate", evaluate_docs)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        lambda s: "rewrite" if s.get("needs_retry") and s.get("retries", 0) < MAX_RETRIES else "generate",
        {"rewrite": "rewrite", "generate": "generate"}
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Create persistent SQLite connection for conversation memory
    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return graph.compile(checkpointer=memory)
