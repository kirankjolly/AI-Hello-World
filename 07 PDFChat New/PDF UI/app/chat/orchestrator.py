"""
Modern Chat Orchestrator - Explicit Application-Controlled Workflow
No agents, no implicit memory, no framework magic.
Full production-ready with cost control and debuggability.
"""
from typing import Iterator, Dict, Any, List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.chat.models import ChatArgs
from app.web.api import (
    get_messages_by_conversation_id,
    add_message_to_conversation
)


@dataclass
class RetrievedDocument:
    """Represents a document chunk retrieved from vector store."""
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class ChatOrchestrator:
    """
    Explicit, application-controlled chat workflow.

    Architecture:
    1. Load conversation history (explicit)
    2. Reformulate question if needed (explicit LLM call)
    3. Retrieve documents (explicit vector search)
    4. Generate answer (explicit LLM call with streaming)
    5. Save to database (explicit)

    No agents, no chains, no implicit behavior.
    """

    def __init__(
        self,
        chat_args: ChatArgs,
        llm: ChatOpenAI,
        retriever,
        max_history_messages: int = 10,
        max_retrieved_docs: int = 5
    ):
        self.chat_args = chat_args
        self.llm = llm
        self.retriever = retriever
        self.max_history_messages = max_history_messages
        self.max_retrieved_docs = max_retrieved_docs

        # Separate LLM for question reformulation (non-streaming, cheaper)
        self.condense_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=False
        )

    def _load_conversation_history(self) -> List[HumanMessage | AIMessage]:
        """
        Explicitly load conversation history from database.
        Application controls how many messages to include.
        """
        messages = get_messages_by_conversation_id(
            self.chat_args.conversation_id
        )

        # Limit history for cost control
        recent_messages = messages[-self.max_history_messages:]

        return [
            HumanMessage(content=msg.content) if msg.role == "human"
            else AIMessage(content=msg.content)
            for msg in recent_messages
        ]

    def _reformulate_question(
        self,
        question: str,
        chat_history: List[HumanMessage | AIMessage]
    ) -> str:
        """
        Explicitly reformulate follow-up questions into standalone questions.

        Example:
        User: "What is LangChain?"
        AI: "LangChain is a framework..."
        User: "How do I use it?"
        → Reformulated: "How do I use LangChain?"
        """
        if not chat_history:
            # First message, no reformulation needed
            return question

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Given a chat history and a follow-up question,
rephrase the follow-up question to be a standalone question that includes all necessary context.

If the question is already standalone, return it as-is."""),
            *chat_history,
            HumanMessage(content=f"Follow-up question: {question}\n\nStandalone question:")
        ])

        try:
            messages = prompt.format_messages()
            response = self.condense_llm.invoke(messages)
            standalone = response.content.strip()

            # Log for debugging
            print(f"[DEBUG] Original: {question}")
            print(f"[DEBUG] Standalone: {standalone}")

            return standalone
        except Exception as e:
            print(f"[ERROR] Question reformulation failed: {e}")
            # Fallback to original question
            return question

    def _retrieve_documents(self, query: str) -> List[RetrievedDocument]:
        """
        Explicitly retrieve relevant documents from vector store.
        Application controls filtering and limits.
        """
        try:
            docs = self.retriever.invoke(query)

            # Limit for cost control
            docs = docs[:self.max_retrieved_docs]

            retrieved = [
                RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=getattr(doc, 'score', 0.0)
                )
                for doc in docs
            ]

            print(f"[DEBUG] Retrieved {len(retrieved)} documents")
            return retrieved

        except Exception as e:
            print(f"[ERROR] Document retrieval failed: {e}")
            return []

    def _format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Include page number if available
            page = doc.metadata.get('page', 'unknown')
            context_parts.append(
                f"[Document {i}, Page {page}]\n{doc.content}"
            )

        return "\n\n".join(context_parts)

    def _build_prompt(
        self,
        question: str,
        context: str,
        chat_history: List[HumanMessage | AIMessage]
    ) -> List[HumanMessage | AIMessage | SystemMessage]:
        """
        Explicitly build the prompt for answer generation.
        Full transparency - no hidden prompts.
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

Guidelines:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Include page references when citing information
- Be concise and accurate"""

        messages = [
            SystemMessage(content=system_prompt),
            *chat_history,
            HumanMessage(content=f"""Context from PDF:
{context}

Question: {question}

Answer:""")
        ]

        return messages

    def _generate_answer(self, messages: List) -> str:
        """
        Explicitly generate answer (non-streaming).
        Single LLM call, full control.
        """
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _generate_answer_streaming(self, messages: List) -> Iterator[str]:
        """
        Explicitly generate answer with streaming.
        Application controls the stream, not the framework.
        """
        try:
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] Streaming failed: {e}")
            yield f"Error: {str(e)}"

    def _save_messages(self, question: str, answer: str):
        """Explicitly save messages to database."""
        add_message_to_conversation(
            conversation_id=self.chat_args.conversation_id,
            role="human",
            content=question
        )
        add_message_to_conversation(
            conversation_id=self.chat_args.conversation_id,
            role="ai",
            content=answer
        )

    def run(self, question: str) -> str:
        """
        Execute the complete workflow (non-streaming).

        Explicit steps:
        1. Load history
        2. Reformulate question
        3. Retrieve documents
        4. Generate answer
        5. Save to DB

        Returns the answer string.
        """
        print(f"[ORCHESTRATOR] Starting workflow for: {question}")

        # Step 1: Load conversation history (explicit)
        chat_history = self._load_conversation_history()
        print(f"[ORCHESTRATOR] Loaded {len(chat_history)} history messages")

        # Step 2: Reformulate question if needed (explicit)
        standalone_question = self._reformulate_question(question, chat_history)

        # Step 3: Retrieve documents (explicit)
        documents = self._retrieve_documents(standalone_question)
        context = self._format_context(documents)

        # Step 4: Build prompt (explicit)
        messages = self._build_prompt(question, context, chat_history)

        # Step 5: Generate answer (explicit)
        answer = self._generate_answer(messages)
        print(f"[ORCHESTRATOR] Generated answer: {len(answer)} chars")

        # Step 6: Save to database (explicit)
        self._save_messages(question, answer)
        print(f"[ORCHESTRATOR] Saved to database")

        return answer

    def stream(self, question: str) -> Iterator[str]:
        """
        Execute the complete workflow with streaming.

        Same explicit steps as run(), but streams the answer.
        Saves after streaming completes.
        """
        print(f"[ORCHESTRATOR] Starting streaming workflow for: {question}")

        # Steps 1-4: Same as run()
        chat_history = self._load_conversation_history()
        standalone_question = self._reformulate_question(question, chat_history)
        documents = self._retrieve_documents(standalone_question)
        context = self._format_context(documents)
        messages = self._build_prompt(question, context, chat_history)

        # Step 5: Stream answer (explicit)
        full_answer = ""
        for chunk in self._generate_answer_streaming(messages):
            full_answer += chunk
            yield chunk

        # Step 6: Save after streaming completes (explicit)
        self._save_messages(question, full_answer)
        print(f"[ORCHESTRATOR] Streaming complete, saved to database")


def build_orchestrator(chat_args: ChatArgs, llm: ChatOpenAI, retriever) -> ChatOrchestrator:
    """
    Factory function to build orchestrator.
    Explicit construction, no framework magic.
    """
    return ChatOrchestrator(
        chat_args=chat_args,
        llm=llm,
        retriever=retriever,
        max_history_messages=10,  # Cost control
        max_retrieved_docs=5      # Cost control
    )
