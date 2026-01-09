from langchain_core.messages import SystemMessage, AIMessage
from llm import get_llm

def generate_answer(state):
    llm = get_llm()
    context = "\n\n".join(d.page_content for d in state["documents"])

    # Build messages with context and conversation history
    system_prompt = [
        SystemMessage(content="Answer the question based on the provided context and conversation history. If the answer is in the context, provide it clearly.")
    ]

    if context:
        system_prompt.append(SystemMessage(content=f"Context:\n{context}"))

    # Combine system prompts with conversation history
    messages = system_prompt + state.get("messages", [])

    # Get AI response
    response = llm.invoke(messages)
    answer = response.content

    # Return answer and add AI message to conversation history
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }
