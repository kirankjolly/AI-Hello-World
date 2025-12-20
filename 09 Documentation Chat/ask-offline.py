from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_llm(query: str):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.load_local("faiss_index_offline", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = Ollama(model="llama3", verbose=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])
