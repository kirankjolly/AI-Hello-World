from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])
