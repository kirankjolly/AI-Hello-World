from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

#WITH LLM TO ANSWER BASED ON CONTEXT FROM SEMANTIC SEARCH

# ---------- 1. Load & split documents ----------
loader = TextLoader("facts.txt")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20
)

docs = loader.load_and_split(text_splitter=text_splitter)

# ---------- 2. Create / load vector DB ----------
embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# ---------- 3. Retrieve relevant chunks ----------
query = "What is an interesting fact about the English language?"

results = db.similarity_search(query, k=3)

context = "\n\n".join([doc.page_content for doc in results])

print("Context for LLM:")
print(context)

# ---------- 4. LLM call (explicit, no chain) ----------
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

messages = [
    SystemMessage(content="Answer using only the provided context."),
    SystemMessage(content=f"Context:\n{context}"),
    HumanMessage(content=query)
]

response = chat.invoke(messages)

print(response.content)
