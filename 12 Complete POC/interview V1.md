# AI Knowledge Assistant — Interview Summary

---

## Full Description

**Overview**

We built an enterprise AI Knowledge Assistant and integrated it into an existing Angular and Java platform. The system covers the full stack — an Angular chat interface for users and an admin portal for document management, a Python FastAPI backend orchestrated with LangGraph, RAG-based document retrieval using Qdrant as the vector store, role-based access control at the retrieval level, multi-layer guardrails including a post-generation hallucination check, conversation memory and rate limiting via Redis, OpenAI for both embeddings and language generation, an OpenAPI 3.0 contract for Angular integration, and the whole thing deployed on Stackit with Kubernetes and load balancing. I can walk through each of those in detail.

**What Is This System?**

We built an AI-powered Knowledge Assistant and integrated it into an existing enterprise platform. The company had an Angular frontend and a Java backend already in place. We introduced a new Python AI backend and connected it to that existing system. Users log into the same Angular portal and now see a chat assistant window where they can ask questions and get answers based on company documents. There is also an admin section in the portal where content managers upload and manage those documents.

**Why RAG?**

A general-purpose language model has no knowledge of your company's internal policies or documents. Using RAG — Retrieval Augmented Generation — we first search the company's document library for the most relevant content, then pass that content to the model along with the user's question. The model answers strictly from what was retrieved, not from its training data. This gives us accurate, company-specific answers instead of guessed or hallucinated ones.

**Architecture**

The Angular frontend sends REST calls to our AI backend over HTTPS. The backend runs as multiple pods on a Kubernetes cluster on Stackit, behind an Ingress load balancer. The pods are stateless — shared state like session data and rate limit counters live in Redis, document metadata in PostgreSQL, and vector embeddings in Qdrant on a persistent volume. The pods call the OpenAI API for both language generation and embedding creation. The existing Java backend was not touched.

**Request Flow**

Every request goes through our FastAPI layer first — JWT validation, rate limiting, then into the LangGraph workflow engine. LangGraph runs the request through an explicit sequence of steps: validate the user and their role, run safety guardrails, then an agent classifies the query and decides whether to use RAG or a structured tool. For RAG, we retrieve permission-filtered document chunks from Qdrant and generate an answer using OpenAI. After generation, a second AI call checks whether the answer is grounded in the retrieved documents. If not, it is replaced with a safe fallback. Finally the response is formatted with citations and returned.

**Permission-Aware Retrieval**

Documents have three access levels — public, manager, and confidential. When querying Qdrant, the user's role is passed as a filter at the database level. A user with a lower role will never receive content from a higher-access document, even if it is semantically relevant to their question.

**Guardrails**

Before any LLM call is made, the query goes through multiple safety checks: empty input rejection, length limits, prompt injection pattern scanning, and off-topic detection. These are lightweight pattern checks that cost nothing. Only queries that pass all checks reach the model. After generation, a second AI call validates that the answer is supported by the retrieved context — catching hallucinations before they reach the user.

**Conversation Memory and Rate Limiting**

Each user session maintains conversation history in Redis, so follow-up questions work naturally without re-stating context. Rate limiting is also tracked in Redis — request counts per user within a time window, enforced consistently across all pods since they all share the same Redis instance.

**Integration with Angular via OpenAPI**

The FastAPI backend auto-generates an OpenAPI 3.0 specification, exposed as a Swagger UI. This gave the Angular team a clear contract for every endpoint — request shapes, response shapes, and error codes — without needing to read backend code. The Angular chat window calls the query endpoint with the user's JWT token. The admin portal calls the document ingestion endpoints. The existing login flow issues the JWT and our backend validates it. No changes were needed in the Java backend or the Angular authentication layer.

**Deployment on Stackit with Kubernetes**

The AI backend runs as multiple replicas on Kubernetes. Kubernetes scales replicas based on load and routes traffic only to healthy pods. Secrets are stored in Kubernetes Secrets and injected as environment variables — nothing sensitive is in the container image. Deployments are rolling — new pods come up healthy before old ones are terminated, giving zero-downtime releases.

**Production Readiness**

Security is enforced at multiple layers: JWT authentication, permission-filtered retrieval at query time, and guardrails against injection and off-topic abuse. Reliability is covered by retry logic with exponential backoff on all LLM calls. Scalability comes from stateless pods and Redis as shared state. Observability is handled by per-step workflow logging and LangSmith tracing for every LLM call. And the hallucination check ensures answers are always tied to real source documents.

---

## Reference Diagrams

### Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ANGULAR FRONTEND                             │
│                                                                      │
│   ┌─────────────────────────┐    ┌──────────────────────────────┐   │
│   │   User Chat Window      │    │   Admin Portal (Doc Upload)  │   │
│   └────────────┬────────────┘    └──────────────┬───────────────┘   │
└────────────────┼──────────────────────────────── ┼──────────────────┘
                 │ HTTPS / REST                    │ HTTPS / REST
                 ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STACKIT — Kubernetes Cluster                      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Ingress / Load Balancer                    │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │                                      │
│          ┌────────────────────┼────────────────────┐                │
│          ▼                    ▼                    ▼                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │  AI Backend  │   │  AI Backend  │   │  AI Backend  │  (Pods)   │
│   │  FastAPI     │   │  FastAPI     │   │  FastAPI     │           │
│   │  (Python)    │   │  (Python)    │   │  (Python)    │           │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│          └───────────────────┼──────────────────┘                   │
│                              │                                       │
│   ┌──────────────────────────▼───────────────────────────────────┐  │
│   │              Shared Services (within cluster)                │  │
│   │                                                              │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │  │
│   │  │   Redis     │  │  PostgreSQL │  │   Qdrant             │ │  │
│   │  │ (Sessions & │  │ (Users, Docs│  │  (Vector Store)      │ │  │
│   │  │ Rate Limits)│  │  Metadata)  │  │                      │ │  │
│   │  └─────────────┘  └─────────────┘  └──────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ (External API calls)
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                      ▼
   ┌──────────────────────────────┐   ┌──────────────────┐
   │  OpenAI API                  │   │  Java Backend    │
   │  (LLM + Embeddings)          │   │  (Existing)      │
   └──────────────────────────────┘   └──────────────────┘
```

### Request Flow Inside the AI Backend

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Layer                                  │
│  — JWT validation, rate limiting (Redis)        │
│  — OpenAPI 3.0 spec auto-generated at /docs     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                 LangGraph Workflow Engine                     │
│                                                              │
│  1. validate_user      — Confirm user, resolve role          │
│         │                                                    │
│  2. apply_guardrails   — Injection check, domain check,      │
│         │                length check                        │
│         │                                                    │
│  3. classify_and_route — Agent decides: RAG or Tool?         │
│         │                                                    │
│    ┌────┴──────────┐                                         │
│    ▼               ▼                                         │
│  RAG Path      Tool Path                                     │
│    │               │                                         │
│  retrieve_docs  execute_tool                                 │
│  (Qdrant,       (structured                                  │
│  permission-    lookup)                                      │
│  filtered)          │                                        │
│  build_context  ────┘                                        │
│    │                                                         │
│  4. generate_answer    — Call Claude with context            │
│         │                                                    │
│  5. validate_grounding — LLM-as-judge hallucination check    │
│         │                                                    │
│  6. format_response    — Add citations, finalize             │
└──────────────────────────────────────────────────────────────┘
                     │
                     ▼
         Answer + Citations → Angular chat window
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Angular |
| AI Backend | Python, FastAPI |
| API Specification | OpenAPI 3.0 (auto-generated) |
| Workflow Orchestration | LangGraph (StateGraph) |
| Language Model | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | Qdrant |
| Session & Rate Limiting | Redis |
| Application Database | PostgreSQL |
| Existing Backend | Java |
| Deployment | Stackit (Kubernetes) |
| Tracing & Observability | LangSmith |
