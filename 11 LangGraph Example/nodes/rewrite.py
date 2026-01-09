from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm

def rewrite_query(state):
    llm = get_llm()
    messages = [
        SystemMessage(content="Rewrite the query for better search."),
        HumanMessage(content=state["question"])
    ]
    rewritten = llm.invoke(messages).content
    return {
        "question": rewritten,
        "retries": state["retries"] + 1
    }
