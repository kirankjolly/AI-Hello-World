"""
Modern Chat Builder - Replaces old chain-based approach
Uses explicit orchestration instead of framework-controlled chains
"""
from app.chat.models import ChatArgs
from app.chat.orchestrator import build_orchestrator
from app.chat.vector_stores.pinecone_modern import build_retriever
from app.chat.llms.modern_llm import llm_map
from app.web.api import (
    set_conversation_components,
    get_conversation_components
)
from app.chat.score import random_component_by_score


def select_component(component_type, component_map, chat_args):
    """
    Select component (LLM/retriever) based on conversation history or scoring.
    Same logic as before, but explicit.
    """
    components = get_conversation_components(chat_args.conversation_id)
    previous_component = components[component_type]

    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    else:
        random_name = random_component_by_score(component_type, component_map)
        builder = component_map[random_name]
        return random_name, builder(chat_args)


def build_chat(chat_args: ChatArgs):
    """
    Modern chat builder - Explicit application-controlled workflow.

    REMOVED:
    - ❌ StreamingConversationalRetrievalChain (black box)
    - ❌ ConversationBufferMemory (implicit memory)
    - ❌ Custom StreamableChain mixin (workaround)
    - ❌ TraceableChain mixin (workaround)

    ADDED:
    - ✅ ChatOrchestrator (explicit workflow control)
    - ✅ Direct LLM streaming (built-in)
    - ✅ Explicit database persistence (no memory wrapper)
    - ✅ Full debuggability and cost control

    Returns:
        ChatOrchestrator with .run() and .stream() methods
    """

    # Select LLM (same component selection logic)
    llm_name, llm = select_component("llm", llm_map, chat_args)

    # Build retriever (explicit, no map needed)
    retriever = build_retriever(
        chat_args,
        k=5  # Explicit cost control
    )

    # Save selected components
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever="pinecone",  # Explicit retriever name
        memory="explicit_db"   # No implicit memory - explicit DB saves
    )

    # Build orchestrator (explicit workflow)
    orchestrator = build_orchestrator(
        chat_args=chat_args,
        llm=llm,
        retriever=retriever
    )

    return orchestrator


# For backward compatibility with conversation_views.py
# The orchestrator has the same .run() and .stream() interface
# as the old chain, so no changes needed in views!
