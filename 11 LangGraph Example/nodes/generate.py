from langchain_core.messages import SystemMessage, AIMessage
from llm import get_llm

def generate_answer(state):
    """
    Generate the final answer using LLM with both context and conversation history.

    This is where the conversation memory is actually USED!

    MAGIC #10: Accessing Conversation History
    ==========================================
    state["messages"] contains ALL previous messages from this conversation!
    - If this is the first message: [HumanMessage("hi")]
    - If this is the 5th message: [HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage]

    LangGraph automatically loaded these from the database based on thread_id!
    """
    llm = get_llm()

    # Build context from retrieved documents
    context = "\n\n".join(d.page_content for d in state["documents"])

    # Create system prompts with instructions
    system_prompt = [
        SystemMessage(content="Answer the question based on the provided context and conversation history. If the answer is in the context, provide it clearly.")
    ]

    # Add document context if available
    if context:
        system_prompt.append(SystemMessage(content=f"Context:\n{context}"))

    # MAGIC #11: Combining Context with Conversation History
    # =======================================================
    # messages array will look like:
    # [
    #   SystemMessage("Answer based on context..."),           ← Instructions
    #   SystemMessage("Context: <documents>"),                 ← Retrieved docs
    #   HumanMessage("who am i"),                              ← Previous question
    #   AIMessage("You are..."),                               ← Previous answer
    #   HumanMessage("what was my previous question"),         ← Current question
    # ]
    #
    # The LLM sees:
    # 1. Instructions (system prompt)
    # 2. Document context (from vector store)
    # 3. Full conversation history (from database via checkpointer)
    #
    # This is why the AI can answer questions like:
    # - "what did I ask before?"
    # - "add 2 more" (remembering previous calculation)
    # - "who did we discuss?" (remembering previous topics)
    messages = system_prompt + state.get("messages", [])

    # Get AI response with full context
    response = llm.invoke(messages)
    answer = response.content

    # MAGIC #12: Updating Conversation History
    # =========================================
    # We return:
    # {
    #   "answer": answer,                       ← For display to user
    #   "messages": [AIMessage(content=answer)] ← Added to conversation history
    # }
    #
    # Because "messages" uses add_messages reducer:
    # - This AIMessage is APPENDED to existing messages
    # - NOT replacing them!
    #
    # After this node:
    # - LangGraph saves a new checkpoint with the updated messages list
    # - Next time this thread_id is used, this AI response will be in state["messages"]
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }
