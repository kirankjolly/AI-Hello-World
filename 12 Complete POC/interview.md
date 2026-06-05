# AI Interview Guide — Three Projects

---

## Project 1: Enterprise AI Knowledge Assistant

**What Is This System?**

We built an AI-powered Knowledge Assistant for an enterprise client with thousands of records, integrated into their existing platform. The company had an Angular frontend and a Java backend already in place. We introduced a new Python AI backend and connected it to that existing system. Users log into the same Angular portal and see a chat assistant window where they can ask questions and get answers grounded in company documents. There is also an admin section in the portal where content managers upload and manage those documents.

**Why RAG?**

A general-purpose language model has no knowledge of your company's internal policies or documents. Using RAG — Retrieval Augmented Generation — we first search the company's document library for the most relevant content, then pass that content to the model along with the user's question. The model answers strictly from what was retrieved, not from its training data. This gives accurate, company-specific answers instead of guessed or hallucinated ones.

**Why Tool Calling Alongside RAG?**

Not every question is answered by searching documents. Some questions require structured data lookups or calculations — things like "calculate my bonus at 10%" or "what is the vacation policy summary?" These are answered faster and more accurately by calling a function than by retrieving paragraphs of text. We use an AI agent to classify each incoming query and route it: document search questions go to the RAG path, calculation and structured lookup questions go to the tool path. Both paths converge at the same answer generation step.

---

**Architecture**

The Angular frontend sends REST calls to our AI backend over HTTPS. The backend runs as multiple pods on a Kubernetes cluster on Stackit, behind an Ingress load balancer. The pods are stateless — shared state like session data and rate limit counters live in Redis, employee and project metadata in PostgreSQL, and vector embeddings in Qdrant on a persistent volume. The pods call the OpenAI API for both embedding creation and language generation. The existing Java backend was not touched.

---

**Authentication and Security**

Every request requires a JWT Bearer token. On login, the user submits their email and password. The backend verifies the password against a bcrypt hash stored in PostgreSQL, then issues a signed JWT containing the user's ID, role, department, and a unique token ID (`jti`) for future revocation support. On every protected request, FastAPI extracts the token, validates the signature and expiry, then re-fetches the employee record from PostgreSQL — this means if an admin deactivates an account or changes a role, the change takes effect on the next request without needing a re-login.

---

**Request Flow**

Every request goes through our FastAPI layer first — JWT validation and rate limiting — then into the LangGraph workflow engine. LangGraph runs the request through an explicit sequence of steps:

1. **validate_user** — confirm user exists and resolve their role from PostgreSQL
2. **apply_guardrails** — lightweight safety checks before any LLM call
3. **classify_and_route** — agent classifies the query and chooses a path
4. **RAG path** — retrieve permission-filtered chunks from Qdrant, build context
5. **Tool path** — execute a company-specific tool, get structured result
6. **generate_answer** — call GPT-4o with the retrieved context or tool result
7. **validate_grounding** — second LLM call checks the answer is grounded in evidence
8. **format_response** — add citations and return

---

**LangGraph as the Workflow Engine**

We chose LangGraph over a simple chain because it gives us an explicit, auditable state machine. Every step is a named node. Every routing decision is a named conditional edge. The state — user details, retrieved chunks, guardrail results, agent decision, generated answer — is typed with a TypedDict and passed explicitly between nodes. This means we can log every step independently, test each node in isolation, and short-circuit to a terminal error node without unwinding the stack. With a black-box agent, if something goes wrong you don't know which step failed. With LangGraph you always know exactly which node produced a bad state.

---

**Permission-Aware Retrieval**

Documents have three access levels — public, manager, and confidential. The user's role maps to a list of allowed levels:

- employee → [public]
- manager → [public, manager]
- admin → [public, manager, confidential]
- hr → [public, manager]

When querying Qdrant, the user's allowed levels are passed as a metadata filter at the database level — a WHERE clause on the access_level field. A lower-role user will never receive a confidential chunk, even if it is semantically the closest match. This is enforced at the vector database level, not the application layer. It is the difference between filtering results after retrieval versus never loading them at all.

---

**Sentence Window Retrieval**

After the vector search returns matched chunks, we expand each match to include its neighboring chunks from the same document. For example, if chunk index 5 matched, we also pull chunks 4 and 6. If two matched chunks are close enough that their windows overlap, we merge them into a single passage rather than sending duplicate text to the LLM. This gives the model a wider, more readable passage while keeping the retrieval signal clean — we search on small, focused chunks, but generate on larger, contextualized passages.

---

**AI Agent and Tool Calling**

The agent uses a lightweight LLM classifier at low temperature to categorize each query into a routing category into one of five routes: `rag`, `calculate`, `policy`, `summarize`, or `list`. Unknown categories fall back to `rag` as a safe default.

For tool-path queries, the agent follows the ReAct pattern — Thought, Action, Observation, Final Answer — making the reasoning auditable:

- **calculate_bonus** — given a salary and a bonus rate, returns gross bonus, estimated tax at 25%, and net bonus
- **lookup_employee_policy** — returns a structured policy summary (vacation, sick leave, remote work, parental leave, performance review, compensation bands) from PostgreSQL
- **summarize_document** — returns a high-level overview of a named company document from PostgreSQL
- **list_available_documents** — returns only documents and policies visible to the current user's role

Every tool has a permission check before execution — the agent verifies the user's role against the document or policy's access level before calling the function. This is the second line of defense after Qdrant's filter: even if a query routes to the tool path instead of RAG, confidential data is never returned to a lower-role user.

---

**Multi-Layer Guardrails**

Before any LLM call is made, the query passes through five sequential checks:

1. **Empty check** — reject blank queries
2. **Length check** — reject queries over 1000 characters (prevents context overflow and hidden instruction attacks)
3. **Prompt injection scan** — regex patterns detect 12 attack types including "ignore previous instructions", "jailbreak", "DAN mode", "reveal your system prompt", "act as", "pretend to be"
4. **Off-domain check** — reject non-work queries (poems, recipes, weather, sports, crypto) to prevent token waste
5. **Post-generation grounding check** — after generation, a second LLM call validates that the answer is supported by the retrieved documents; if not, it is replaced with a safe fallback

The cheapest checks run first so the expensive LLM calls are only reached by clean queries. The grounding check fails open — if the judge itself errors, the original answer is passed through rather than blocking a valid response.

---

**Conversation Memory and Rate Limiting**

Each user session maintains conversation history in Redis (sessions and messages tables), so follow-up questions work naturally without re-stating context. The session is created on the first query and reused across subsequent turns via a session_id returned to the client.

Rate limiting uses a sliding window counter algorithm also backed by Redis — request counts per user within a configurable time window, enforced consistently across all pods since they all share the same Redis instance. A per-user counter increments on each request and triggers an HTTP 429 when the limit is reached.

---

**File Ingestion Pipeline**

The admin portal supports PDF and TXT uploads via a `/upload_document` endpoint. The backend extracts text using PyPDFLoader for PDFs and plain UTF-8 read for text files. The extracted text is then split using `RecursiveCharacterTextSplitter` — which tries to split on paragraph breaks, then newlines, then sentence boundaries, then words, never mid-word — with a configurable chunk size and overlap. Each chunk is embedded using OpenAI's text-embedding model and stored in Qdrant with metadata: doc_id, title, department, access_level, and chunk_index. The chunk_index is what enables sentence window retrieval later.

---

**Integration with Angular via OpenAPI**

FastAPI auto-generates an OpenAPI 3.0 specification exposed as a Swagger UI. This gave the Angular team a clear contract for every endpoint — request shapes, response shapes, and error codes — without needing to read backend code. The Angular chat window calls the `/ask` endpoint with the user's JWT token. The admin portal calls the document upload endpoints. The existing login flow issues the JWT and our backend validates it. No changes were needed in the Java backend or the Angular authentication layer.

---

**Deployment on Stackit with Kubernetes**

The AI backend runs as multiple replicas on Kubernetes. Kubernetes scales replicas based on load and routes traffic only to healthy pods via the `/health` endpoint. Secrets — JWT secret, API keys — are stored in Kubernetes Secrets and injected as environment variables at runtime; nothing sensitive is in the container image. Deployments are rolling — new pods come up healthy before old ones are terminated, giving zero-downtime releases. Pod statelessness is critical here: because sessions, rate limits, and document metadata all live in external services (Redis, PostgreSQL, Qdrant), any pod can handle any request.

---

**Production Readiness**

Security is enforced at multiple layers: JWT authentication with bcrypt, permission-filtered retrieval at the vector database level, agent-level permission re-verification before tool execution, and guardrails against injection and off-topic abuse. Reliability is covered by retry logic with exponential backoff on all LLM calls via `tenacity`. Scalability comes from stateless pods and Redis as shared state. Observability is handled by per-step workflow logging and LangSmith tracing for every LLM call. The hallucination check ensures answers are always tied to real source documents.

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
│   │  │ (Sessions & │  │ (Employees, │  │  (Vector Store)      │ │  │
│   │  │ Rate Limits)│  │  Projects,  │  │  Permission-filtered │ │  │
│   │  │             │  │  Tool Data) │  │  embeddings          │ │  │
│   │  └─────────────┘  └─────────────┘  └──────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ (External API calls)
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                      ▼
   ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐
   │          OpenAI API (Embeddings + Generation)          │  │  Java Backend    │
   │          GPT-4o · text-embedding-ada-002              │  │  (Existing)      │
   └─────────────────┘  └────────────────┘  └──────────────────┘
```

---

### Request Flow Inside the AI Backend

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Layer                                  │
│  — JWT validation (python-jose, bcrypt)         │
│  — Rate limiting: sliding window (Redis)        │
│  — OpenAPI 3.0 spec auto-generated at /docs     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                 LangGraph Workflow Engine                     │
│                 TypedDict State, explicit edges               │
│                                                              │
│  1. validate_user      — Confirm user, resolve role          │
│         │                (PostgreSQL lookup)                 │
│  2. apply_guardrails   — Empty, length, injection,           │
│         │                off-domain checks                   │
│         │                                                    │
│  3. classify_and_route — Agent LLM decides: RAG or Tool?     │
│         │                                                    │
│    ┌────┴──────────────────────────────────────┐             │
│    ▼  RAG Path                Tool Path        │             │
│    │                          │                │             │
│  retrieve_docs            execute_tool         │             │
│  (Qdrant,                 ┌──────────────┐     │             │
│  WHERE access_level       │ calculate_   │     │             │
│  IN allowed_levels)       │ bonus        │     │             │
│    │                      │ lookup_      │     │             │
│  sentence window          │ policy       │     │             │
│  expansion                │ summarize_   │     │             │
│    │                      │ document     │     │             │
│  build_context            │ list_docs    │     │             │
│    │                      └──────┬───────┘     │             │
│    └──────────────────────────── ┼ ────────────┘             │
│                                  │                           │
│  4. generate_answer    — GPT-4o with context or tool result  │
│         │                                                    │
│  5. validate_grounding — LLM-as-judge hallucination check    │
│         │                                                    │
│  6. format_response    — Add citations, finalize             │
└──────────────────────────────────────────────────────────────┘
                     │
                     ▼
         Answer + Citations → Angular chat window
         Session persisted to Redis
```

---

### Permission Model

```
User Role      →  Allowed Access Levels  →  Visible Data
─────────────────────────────────────────────────────────
employee       →  [public]               →  HR handbook, IT security
manager        →  [public, manager]      →  + Finance policy, compensation bands
admin          →  [public, manager,      →  + Executive compensation
                   confidential]
hr             →  [public, manager]      →  Same as manager
```

Permission enforcement layers:
1. **Qdrant WHERE filter** at query time — confidential chunks never loaded into memory
2. **Agent tool check** before every tool execution — role verified against policy/document access_level
3. **`list_available_documents`** filters by allowed levels — prevents document existence leakage

---

### Tool Calling — ReAct Loop

```
User: "What is my bonus if my salary is $80,000 at 12%?"
      │
      ▼
classify_and_route → "calculate"
      │
      ▼
Agent Thought:  "User wants a bonus calculation. I should call
                calculate_bonus, not guess the answer."
Agent Action:   calculate_bonus(salary=80000, bonus_rate=0.12)
Agent Obs:      Bonus Calculation:
                  Annual Salary:    $80,000.00
                  Bonus Rate:       12.0%
                  Gross Bonus:      $9,600.00
                  Est. Tax (25%):   $2,400.00
                  Net Bonus (est.): $7,200.00
Agent Thought:  "I have the result. I can now answer."
Final Answer:   "Based on your salary of $80,000 with a 12% bonus
                rate, your gross bonus is $9,600 and your estimated
                net bonus after tax is $7,200."
```

---

### Database Schema

```
PostgreSQL
├── employees
│   ├── employee_id    TEXT UNIQUE       — e.g. "emp_001"
│   ├── name, email    TEXT              — identity
│   ├── department     TEXT              — HR, Finance, IT...
│   ├── role           TEXT              — employee|manager|admin|hr
│   ├── joining_date   TEXT              — YYYY-MM-DD
│   ├── password_hash  TEXT              — bcrypt
│   └── is_active      INTEGER           — soft delete flag
│
└── projects
    ├── name           TEXT
    ├── start_date     TEXT
    ├── end_date       TEXT (nullable)   — NULL = ongoing
    ├── status         TEXT              — active|completed|on-hold
    └── assignee_id    FK → employees

Redis
├── sessions           — session_id → {user_id, title, messages[]}
│   └── messages       — ordered list of {role, content, created_at}
└── rate_limit_log     — user_id → sliding window request timestamps
```

---

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Angular |
| AI Backend | Python, FastAPI |
| API Specification | OpenAPI 3.0 (auto-generated by FastAPI) |
| Workflow Orchestration | LangGraph (StateGraph + TypedDict state) |
| AI Agent Pattern | ReAct (Thought → Action → Observation) |
| Language Model | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding model |
| Vector Database | Qdrant (permission-filtered at query level) |
| Tool Calling | LangChain `@tool` + LLM classifier |
| Authentication | JWT (python-jose) + bcrypt (passlib) |
| Session Memory | Redis (conversation history, multi-turn) |
| Rate Limiting | Redis (sliding window counter) |
| Application Database | PostgreSQL (employees, projects, tool data) |
| File Processing | PyPDFLoader (PDF), UTF-8 (TXT) |
| Text Splitting | RecursiveCharacterTextSplitter (LangChain) |
| Retry Logic | tenacity (exponential backoff on LLM calls) |
| Existing Backend | Java (untouched) |
| Deployment | Stackit (Kubernetes, rolling deploys) |
| Tracing & Observability | LangSmith (per-LLM-call distributed tracing) |

---

## Key Interview Talking Points

**On LangGraph vs. simple chains:**
"A chain runs linearly with no visibility. LangGraph gives you a named state machine — every step is a node, every routing decision is a conditional edge, the state is typed. When something fails in production you know exactly which node produced bad state. And you can short-circuit to a terminal error node cleanly without unwinding."

**On permission filtering at the DB level:**
"There is a fundamental difference between filtering results after retrieval and never loading them at all. We pass the user's allowed access levels as a WHERE clause to Qdrant. A confidential chunk never enters memory for a lower-role user. That is the correct place to enforce it."

**On tool calling vs. RAG:**
"The routing decision is: is this question answered by searching unstructured text, or by executing a function? 'How many vacation days do I get?' — that is a paragraph in a document, RAG is right. 'Calculate my bonus at 10%' — that is math, a tool is right. Sending a calculation question to RAG gets you a quoted policy paragraph, not a number. The agent makes that classification for every query."

**On hallucination check:**
"After generation, we make a second LLM call — the judge — and ask it to evaluate whether the answer is actually supported by the retrieved context. We give it the context, the answer, and ask for a yes/no grounding verdict. If it says no, we replace the answer with a safe fallback before it reaches the user. The judge fails open — if the check itself errors, we pass the answer through rather than blocking a valid response."

**On stateless pods:**
"We can add or remove pods freely because no state lives in the pod. Sessions are in Redis. Rate limit counters are in Redis. Document metadata is in PostgreSQL. Embeddings are in Qdrant. Every pod can handle every request. This is what makes horizontal scaling work correctly."

**On the JWT re-fetch pattern:**
"We decode the JWT to get the user_id, but then we re-fetch the employee record from PostgreSQL on every request. This means role changes and account deactivations take effect immediately — we are not trusting a stale claim baked into a token that might have 30 days left on it."

---

## If I Were to Extend This — Future Improvements

*Use this section after walking through the project. It shows you understand the system's current limits and where the real engineering work would go next.*

---

**On retrieval quality — Hybrid Search:**
"One thing I am not fully happy with is that we only use dense vector search right now. Vector similarity is great for semantic meaning, but it misses exact matches — things like an employee ID, a project code, or a specific clause number. Qdrant actually supports sparse vectors alongside dense, so you can run BM25 keyword search in parallel and merge the two result sets using Reciprocal Rank Fusion. That would cover both the 'find me anything related to parental leave' queries and the 'find policy clause 4.2.1' queries. It is on my list as the first retrieval improvement I would ship."

---

**On retrieval quality — Re-ranking:**
"The other retrieval improvement I would add is a cross-encoder re-ranker after the initial vector search. Right now we retrieve the top-K chunks by embedding similarity and pass them straight to the LLM. A re-ranker like Cohere Rerank or a local ms-marco model takes those K candidates and scores each one against the actual query text — not just the embedding — and reorders them. The LLM then sees the most relevant chunk first rather than the most geometrically close one. It is a relatively cheap step that noticeably improves answer quality on complex questions."

---

**On cost — Semantic Caching:**
"If you look at real enterprise usage, a large chunk of queries are repeats. 'What is the vacation policy?' gets asked by someone new every week. Right now we hit the LLM every single time. A semantic cache stores previous query embeddings alongside their answers. On a new request, if the embedding is close enough to something we have already answered, we return the cached answer and skip the LLM call entirely. The cost savings on a real deployment would be significant, and the latency for those users drops to near zero."

---

**On cost — Model Tiering:**
"Not every step in our pipeline needs GPT-4o. The guardrail checks — is this query empty, is it an injection attempt, is it off-domain — are simple pattern checks that do not need a powerful model at all. The classifier that decides RAG vs. tool path is also a straightforward categorisation task. I would push those steps to GPT-4o-mini or even a fine-tuned smaller model. GPT-4o only gets called for the actual answer generation and the grounding check. That change alone would probably cut our per-request LLM cost by 60 to 70 percent without touching answer quality."

---

**On user experience — Streaming:**
"Right now the user submits a question and stares at a blank screen until the full answer is ready. For a three-sentence answer that is fine, but for a longer policy explanation it feels slow. The fix is straightforward — FastAPI supports Server-Sent Events natively and LangChain supports streaming callbacks. You stream tokens to the Angular frontend as they come out of the model, the same way ChatGPT does it. It does not make the system faster, but it makes it feel faster, which matters for adoption."

---

**On scalability — Async Ingestion:**
"The document upload endpoint right now is synchronous. The HTTP request sits open while we parse the PDF, split it into chunks, call OpenAI for embeddings on every chunk, and write to Qdrant. For a small document that is fine. For a 200-page PDF that times out. The right architecture is a task queue — the upload endpoint accepts the file, queues a Celery job backed by Redis, and immediately returns a job ID to the admin. The worker processes it in the background. The admin portal polls a status endpoint to see when it is done. That is how you handle large document ingestion in production."

---

**On complex questions — GraphRAG:**
"There is a class of question our current system handles poorly: relational questions that cross multiple documents and data sources. Something like 'who is the finance manager and what projects are they leading and what is their travel policy?' — that cuts across the employee table, the projects table, and a policy document. Flat chunk retrieval does not handle that well. GraphRAG builds a knowledge graph on top of the documents, where entities and relationships are nodes and edges, and uses graph traversal alongside vector retrieval to answer multi-hop questions. That is the architectural direction I would go if the query complexity grew beyond what single-document lookup can handle."

---

**Closing line to use in the interview:**
"If I had to pick three things to ship next: hybrid search because we are leaving recall on the table with dense-only retrieval, semantic caching because it is free money on repeated queries, and async ingestion because the synchronous upload will be the first thing that breaks in production. Everything else — re-ranking, streaming, GraphRAG — is the roadmap beyond that."

---

---

## Project 2: AI-Guided Service Discovery Assistant

**What Is This System?**

This project is about making structured business data explorable through conversation. The client had service data — organised as a three-level hierarchy: Sectors, then Buckets within each sector, then individual Services within each bucket — stored in SQL databases and exported as Excel files. Instead of navigating a dropdown-heavy UI or writing queries, users can now find what they need by just describing what they are looking for in plain language.

**How the Conversation Flow Works**

The agent does not dump a flat list at the user. It guides them through the hierarchy step by step. A user starts with an open-ended request — something like "I need a service related to financial compliance." The agent classifies the intent and first narrows it to the relevant Sector. The UI then surfaces that sector. Once the user confirms or selects, the agent moves to the Bucket level and presents the relevant options there. Only after the bucket is confirmed does the agent present the actual Services. This multi-step guided flow is intentional — it mirrors how a knowledgeable human consultant would help someone navigate a large service catalogue rather than overwhelming them with everything at once.

**Why LangGraph for This**

We used LangGraph because the conversation has explicit state — which level of the hierarchy the user is currently at, what they have already selected, and what the next valid step is. A simple chain cannot track that. With LangGraph we have a named node for each stage of the flow — classify intent, present sectors, confirm sector, present buckets, confirm bucket, present services — and conditional edges that determine which node fires next based on the current state. If the user backtracks or changes their mind, the state handles it cleanly.

**The Data Layer**

The Excel files exported from the SQL database are ingested at startup and indexed. The agent queries this indexed data at each stage of the conversation rather than calling the LLM to guess the answer. The LLM handles natural language understanding and intent classification. The structured service data always comes from the source, not from model memory.

**Deployment**

Deployed on Stackit, same infrastructure pattern as the Knowledge Assistant — stateless pods behind an ingress load balancer.

---

**Key Talking Points**

**On why not just a search box:**
"A search box gives you results but no guidance. If you do not know the terminology of the service catalogue, you will search for the wrong thing and get nothing useful. The agent asks clarifying questions the way a human expert would — it meets the user where they are rather than expecting them to already know the structure of the data."

**On the three-level hierarchy in state:**
"Sectors, Buckets, Services — that hierarchy is the business model. The LangGraph state tracks exactly where the user is in that hierarchy at every turn. Each level is a node. The edges are conditional on what the user selected or clarified. This is why we chose LangGraph over a simple chain — the conversation has real branching state, not just a linear sequence of LLM calls."

**On LLM vs. structured data:**
"The LLM does one thing in this system: it understands what the user means. The actual service data — names, descriptions, availability — always comes from the indexed source data. We never ask the LLM to recall a service name. That is how you avoid hallucination in a catalogue system."

---

---

## Project 3: MCP Server for Salesforce Integration

**What Is This System?**

Salesforce had a requirement that their AI agents needed to call external services through tools rather than raw REST APIs. The MCP — Model Context Protocol — is Anthropic's open standard for exposing tools to AI models in a structured, discoverable way. We built a Java MCP server, hosted on Heroku, that wraps the client's real backend API services as MCP tools. Salesforce's AI layer connects to this server and calls those tools by name, passing structured parameters, without ever touching the raw API directly.

**Why MCP Instead of Just Calling the APIs Directly**

Salesforce's direction is to move away from hardcoded API integrations inside their platform. With MCP, the tools are self-describing — each tool has a name, a description, and a typed parameter schema. The AI agent in Salesforce can discover what tools are available, understand what each one does, and call the right one based on the user's intent. That is fundamentally different from a static API integration where a developer has to hardcode which endpoint to call. MCP makes the integration dynamic and AI-navigable.

**The Java MCP Server**

We built the server in Java. Each tool handler wraps one real API service call — authentication, parameter mapping, error handling — and exposes it through the MCP protocol. The server runs on Heroku and is always-on, so Salesforce can reach it whenever its agents need to. We used Claude via Claude Desktop to test the tool connections during development — you point Claude at the MCP server and ask it to perform tasks, and it calls the tools the same way Salesforce will in production.

**What This Means Architecturally**

This pattern is significant beyond just this project. MCP is becoming the standard way AI agents connect to external systems — the same way REST became the standard for web APIs. By building an MCP server you are not just integrating with one AI platform. Any MCP-compatible agent — Claude, Salesforce AgentForce, or future platforms — can connect to the same server and use the same tools without any changes on the server side.

---

**Key Talking Points**

**On why MCP over a direct API:**
"Salesforce's requirement was tools, not APIs. The difference is that a tool is self-describing — it tells the AI what it does, what parameters it accepts, and what it returns. The AI can then decide at runtime which tool to call based on the user's request. A hardcoded API integration cannot do that. MCP is what makes the integration AI-navigable rather than developer-hardcoded."

**On the Java implementation:**
"We wrote the server in Java because that is what the team was already running for their backend services. The MCP protocol is transport-agnostic — it runs over standard I/O or HTTP. Each tool is a handler class that takes structured input, calls the real service, and returns a structured result. The AI never sees the raw HTTP layer."

**On using Claude for testing:**
"During development we pointed Claude Desktop at the MCP server to validate the tools before Salesforce connected. Claude would call the tools the same way Salesforce would — by reading the tool descriptions and deciding which one to invoke. If Claude could complete the task correctly using the tools, we knew the server was ready. That is a very efficient way to test an MCP server."

**On the broader significance:**
"MCP is still early but it is moving fast. The pattern of wrapping existing services as MCP tools rather than rebuilding them is how most enterprise AI integrations will work going forward. You keep your existing backend untouched and expose it as tools that any AI agent can discover and call. That is the same principle as REST APIs but designed for AI agents rather than human developers."
