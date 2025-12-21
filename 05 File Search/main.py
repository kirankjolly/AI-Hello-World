from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
load_dotenv()

#Without LLM, SEMANTIC SEARCH ONLY
embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=10
)

loader = TextLoader("facts.txt")

docs = loader.load_and_split(text_splitter=text_splitter)

db=Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

results = db.similarity_search("What is an interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)