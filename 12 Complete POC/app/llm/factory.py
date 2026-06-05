"""
app/llm/factory.py — LLM Instance Factory

Centralised factory so guardrails, retriever, and agents all share
the same LLM construction logic without circular imports.
"""

from app.config import LLM_PROVIDER, LLM_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY


def get_llm(temperature: float = 0.0):
    """
    Return the configured LLM instance.

    LLM_PROVIDER controls which backend is used:
      "anthropic" → Claude (claude-sonnet-4-6)
      "openai"    → GPT (default gpt-4o-mini)
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL or "claude-sonnet-4-6",
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL or "gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,
            temperature=temperature,
        )
