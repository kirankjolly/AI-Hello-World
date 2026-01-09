from vectorstore import get_vectorstore

def retrieve_docs(state):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    return {"documents": docs}
