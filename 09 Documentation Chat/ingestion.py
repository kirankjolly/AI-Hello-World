from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from custom_loader import UTF8BSHTMLLoader

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = DirectoryLoader(
        "langchain-docs/api.python.langchain.com/en/latest/",
        glob="**/*.html",
        loader_cls=UTF8BSHTMLLoader,
        loader_kwargs={"bs_kwargs": {"features": "html.parser"}}
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to FAISS")
    db = FAISS.from_documents(documents, embeddings)

    db.save_local("faiss_index")
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
