"""
Modern LLM Configuration (2024-2025)
Uses: langchain-openai with modern patterns
"""
import os
from langchain_openai import ChatOpenAI


def build_llm(chat_args, model_name: str = "gpt-4"):
    """
    Explicitly build LLM with cost controls and streaming support.

    Args:
        chat_args: Chat arguments containing streaming flag
        model_name: Model to use (default: gpt-4)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model_name,
        temperature=0.7,
        streaming=chat_args.streaming,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # Cost control: limit output tokens
        max_tokens=2000,
        # Timeout for production
        request_timeout=60
    )


# LLM map for component selection (compatible with existing code)
llm_map = {
    "gpt-4": lambda chat_args: build_llm(chat_args, "gpt-4"),
    "gpt-4-turbo": lambda chat_args: build_llm(chat_args, "gpt-4-turbo-preview"),
    "gpt-3.5-turbo": lambda chat_args: build_llm(chat_args, "gpt-3.5-turbo"),
}
