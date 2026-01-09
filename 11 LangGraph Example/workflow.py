from graph import build_graph
from ingest import ingest_pdfs
from langchain_core.messages import HumanMessage

# Run PDF ingestion on startup
print("Checking for new PDFs to ingest...")
ingest_pdfs()
print("Building graph...")

app = build_graph()

def ask(question: str, thread_id: str = "default"):
    """
    Ask a question with conversation memory.

    Args:
        question: The user's question
        thread_id: Conversation thread ID (use same ID to continue a conversation)
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "retries": 0
    }, config)

    return result["answer"]
