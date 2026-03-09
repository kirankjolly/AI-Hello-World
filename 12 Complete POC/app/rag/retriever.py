"""
app/rag/retriever.py — Permission-Aware Retriever + RAG Pipeline

[Concept: RAG Pipeline + Security - Permission Aware RAG]

────────────────────────────────────────────────────────────────
HOW RAG REDUCES HALLUCINATION
────────────────────────────────────────────────────────────────
A bare LLM like Claude or GPT-4 is trained on data up to a
cutoff date. It has no knowledge of YOUR company's documents.

Without RAG:
  Q: "What is our parental leave policy?"
  A: [Hallucinated generic answer, or "I don't know"]

With RAG:
  1. We retrieve the actual HR handbook chunk about parental leave
  2. We inject it into the LLM's prompt as CONTEXT
  3. We instruct the LLM: "Answer ONLY from the provided context"
  4. The LLM reads YOUR document and answers accurately

Key insight: The LLM isn't "knowing" the answer — it's READING
the answer from your documents and rephrasing it.

This is why RAG works:
  - LLM hallucination: inventing facts from its training data
  - RAG grounded answer: reading facts from provided context

The system prompt enforces this by saying:
  "If the answer is not in the context, say 'I don't know'"
────────────────────────────────────────────────────────────────
"""

from typing import List, Tuple, Optional, NamedTuple
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_MODEL, LLM_PROVIDER, TOP_K_RESULTS, WINDOW_SIZE
from app.models.schemas import UserRole, RetrievedChunk, Citation
from app.security.permissions import get_allowed_access_levels
from app.vector_store.chroma_store import vector_store
from app.observability.logger import (
    log_retrieval, log_llm_call, log_workflow_step, logger
)


# ──────────────────────────────────────────────
# Internal Types
# ──────────────────────────────────────────────

class _IndexedChunk(NamedTuple):
    """
    Pairs a RetrievedChunk with its chunk_index from ChromaDB metadata.

    chunk_index is intentionally absent from the public RetrievedChunk schema
    (schemas.py) — it is an internal storage detail. This type carries it
    only within the retrieval pipeline so expand_chunks_with_window can group
    and sort matched chunks by their position inside the source document.
    """
    chunk_index: int
    chunk: RetrievedChunk


# ──────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────

def get_llm():
    """
    Return a configured LLM instance based on LLM_PROVIDER setting.

    LLM_PROVIDER=openai    → ChatOpenAI  (only OPENAI_API_KEY needed)
    LLM_PROVIDER=anthropic → ChatAnthropic (ANTHROPIC_API_KEY needed)
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
        )
    else:  # default: openai
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1024,
        )


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

def retrieve_documents(
    query: str,
    user_role: UserRole,
    user_id: str,
    k: int = TOP_K_RESULTS,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant document chunks for a query,
    filtered to only include documents the user is allowed to see.

    Step 1: Determine which access levels the user can see
    Step 2: Query ChromaDB with permission filter + semantic similarity
    Step 3: Return structured RetrievedChunk objects
    """
    log_workflow_step("retrieve_documents", user_id, f"query='{query[:60]}'")

    # Step 1: Permission-aware access level list
    # e.g., employee → ["public"]
    # e.g., manager  → ["public", "manager"]
    allowed_levels = get_allowed_access_levels(user_role)

    log_workflow_step(
        "permission_filter", user_id,
        f"role={user_role.value} | allowed_levels={allowed_levels}"
    )

    # Step 2: Vector similarity search with permission filter
    raw_results = vector_store.similarity_search(
        query=query,
        allowed_access_levels=allowed_levels,
        k=k,
    )

    # Step 3: Convert to typed objects, preserving chunk_index for window expansion
    chunks:  List[RetrievedChunk] = []
    indexed: List[_IndexedChunk]  = []
    scores:  List[float]          = []

    for result in raw_results:
        meta = result["metadata"]
        chunk = RetrievedChunk(
            doc_id=meta.get("doc_id", "unknown"),
            title=meta.get("title", "Unknown"),
            department=meta.get("department", "Unknown"),
            access_level=meta.get("access_level", "public"),
            content=result["content"],
            score=result["score"],
        )
        chunks.append(chunk)
        indexed.append(_IndexedChunk(chunk_index=meta.get("chunk_index", 0), chunk=chunk))
        scores.append(result["score"])

    # Step 4: Sentence window expansion (skip if WINDOW_SIZE=0)
    # Expands each matched chunk to include neighboring chunks from the same
    # document, giving the LLM a wider passage without affecting retrieval ranking.
    if WINDOW_SIZE > 0:
        chunks = expand_chunks_with_window(indexed, allowed_levels, WINDOW_SIZE)
        scores = [c.score for c in chunks]

    log_retrieval(user_id, query, len(chunks), scores)

    return chunks


# ──────────────────────────────────────────────
# Sentence Window Expansion
# ──────────────────────────────────────────────

def expand_chunks_with_window(
    indexed_chunks: List[_IndexedChunk],
    allowed_levels: List[str],
    window_size: int,
) -> List[RetrievedChunk]:
    """
    Expand matched chunks by fetching their neighbors, merging overlapping
    windows into single passages to avoid duplicate text sent to the LLM.

    ────────────────────────────────────────────────────────────────
    GROUPING AND DEDUPLICATION ALGORITHM
    ────────────────────────────────────────────────────────────────
    Two matched chunks from the same document whose windows overlap
    must be merged into one passage. Windows overlap when:
        |idx_a - idx_b| <= 2 * window_size

    Example (window_size=1):
        Chunk A idx=5 → window [4, 5, 6]
        Chunk B idx=6 → window [5, 6, 7]
        |5 - 6| = 1 <= 2  →  overlap → merge to [4, 5, 6, 7]

    Algorithm:
      1. Sort matched chunks by (doc_id, chunk_index)
      2. Greedy merge: extend current group while same doc and gap ≤ 2*W
      3. For each group: fetch full index range from ChromaDB, join content
      4. Score = highest score in the group (anchor chunk)
      5. Sort output by score descending
    ────────────────────────────────────────────────────────────────

    Args:
        indexed_chunks: _IndexedChunk list from retrieve_documents Step 3
        allowed_levels: Permission filter passed through to fetch_neighbors
        window_size:    Neighbors on each side to include (from WINDOW_SIZE)

    Returns:
        List[RetrievedChunk] with expanded content, ordered by score desc.
    """
    if not indexed_chunks or window_size == 0:
        return [ic.chunk for ic in indexed_chunks]

    # Step 1: Sort by (doc_id, chunk_index) for greedy interval merging
    sorted_items = sorted(indexed_chunks, key=lambda ic: (ic.chunk.doc_id, ic.chunk_index))

    # Step 2: Greedy merge — build groups of overlapping windows
    groups: List[List[_IndexedChunk]] = []

    for item in sorted_items:
        if not groups:
            groups.append([item])
            continue

        last_group = groups[-1]
        last_item  = last_group[-1]

        same_doc    = item.chunk.doc_id == last_item.chunk.doc_id
        gap         = item.chunk_index - last_item.chunk_index
        overlapping = gap <= 2 * window_size

        if same_doc and overlapping:
            last_group.append(item)
        else:
            groups.append([item])

    # Step 3: For each group, fetch expanded content and build one RetrievedChunk
    expanded: List[RetrievedChunk] = []

    for group in groups:
        # Anchor = highest-scoring matched chunk in this group
        anchor_item = max(group, key=lambda ic: ic.chunk.score)
        anchor      = anchor_item.chunk

        # Full index range: span of all group members ± window_size
        min_idx = min(ic.chunk_index for ic in group)
        max_idx = max(ic.chunk_index for ic in group)
        start   = max(0, min_idx - window_size)   # clamp: no negative indices
        end     = max_idx + window_size            # ChromaDB handles past-end gracefully

        # Fetch neighbors: direct metadata lookup, not semantic search
        neighbors = vector_store.fetch_neighbors(
            doc_id=anchor.doc_id,
            start_index=start,
            end_index=end,
            allowed_access_levels=allowed_levels,
        )

        if neighbors:
            # Join in document order (already sorted by chunk_index in fetch_neighbors)
            expanded_content = "\n\n".join(content for _, content in neighbors)
        else:
            # Fallback: use original matched content if neighbor fetch failed
            expanded_content = anchor.content
            logger.warning(
                f"[RETRIEVER] fetch_neighbors returned empty for "
                f"doc_id={anchor.doc_id} range=[{start},{end}] — using original chunk"
            )

        expanded.append(RetrievedChunk(
            doc_id=anchor.doc_id,
            title=anchor.title,
            department=anchor.department,
            access_level=anchor.access_level,
            content=expanded_content,
            score=anchor.score,   # score of the anchor, not the neighbors
        ))

    # Step 4: Re-sort by score descending (most relevant expanded block first)
    expanded.sort(key=lambda c: c.score, reverse=True)
    return expanded


# ──────────────────────────────────────────────
# Context Building
# ──────────────────────────────────────────────

def build_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is labeled with its source document so the LLM can
    attribute information to specific documents (enabling citations).

    Example output:
        [Source 1] HR Handbook (HR Department):
        Full-time employees receive 15 vacation days per year...

        [Source 2] Vacation Policy 2024 (HR Department):
        Vacation days accrue at 1.25 days per month...
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk.title} ({chunk.department} Department):\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(context_parts)


# ──────────────────────────────────────────────
# RAG Prompt
# ──────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an AI assistant for a company's internal knowledge base.

Your job is to answer employee questions based ONLY on the provided document context below.

IMPORTANT RULES:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If the answer is not in the context, say exactly: "I don't have information about this in the available documents."
3. Always mention which document(s) your answer comes from (e.g., "According to the HR Handbook...").
4. Be concise and factual. Do not add opinions or speculation.
5. If the context contains conflicting information, mention both versions.

This ensures your answers are grounded in actual company documents, not assumptions."""


# ──────────────────────────────────────────────
# RAG Answer Generation
# ──────────────────────────────────────────────

def generate_rag_answer(
    query: str,
    chunks: List[RetrievedChunk],
    user_id: str,
) -> Tuple[str, List[Citation]]:
    """
    Generate a grounded answer using RAG.

    Pipeline:
      1. Build context string from retrieved chunks
      2. Construct the prompt: system rules + context + user question
      3. Call the LLM
      4. Extract citations from the retrieved chunks

    The system prompt enforces "answer only from context" — this
    is the guardrail against hallucination.

    Returns:
        (answer_text, list_of_citations)
    """
    log_workflow_step("generate_rag_answer", user_id)

    # Step 1: Build context
    context = build_context(chunks)

    # Step 2: Construct the user message (context + question)
    user_message = f"""Here are the relevant documents from our knowledge base:

{context}

---

Question: {query}

Please answer based only on the documents above."""

    # Log token estimate (rough: 1 token ≈ 4 chars)
    log_llm_call(user_id, LLM_MODEL, len(user_message) // 4, len(context))

    # Step 3: Call LLM
    llm = get_llm()
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    answer = response.content

    # Step 4: Build citations from retrieved chunks
    citations = []
    seen_docs = set()
    for chunk in chunks:
        if chunk.doc_id not in seen_docs and chunk.score > 0.3:
            # Only cite documents with a meaningful relevance score
            citations.append(Citation(
                doc_id=chunk.doc_id,
                title=chunk.title,
                department=chunk.department,
                snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            ))
            seen_docs.add(chunk.doc_id)

    logger.info(f"[RAG] Generated answer ({len(answer)} chars) with {len(citations)} citations")

    return answer, citations
