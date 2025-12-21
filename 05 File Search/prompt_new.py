from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from redundant_filter_retriever_new import RedundantFilterRetriever
from dotenv import load_dotenv

load_dotenv()

# ---------- Models ----------
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

embeddings = OpenAIEmbeddings()

# ---------- Vector DB ----------
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# ---------- Retriever (your custom logic stays!) ----------
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

# ---------- Step 1: Retrieve ----------
query = "What is an interesting fact about the English language?"

docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in docs)

# ---------- Step 2: Prompt ----------
messages = [
    SystemMessage(content="Answer using only the provided context."),
    SystemMessage(content=f"Context:\n{context}"),
    HumanMessage(content=query)
]

# ---------- Step 3: LLM ----------
response = chat.invoke(messages)

print(response.content)
