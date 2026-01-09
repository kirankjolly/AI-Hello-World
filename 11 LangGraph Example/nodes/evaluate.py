def evaluate_docs(state):
    if not state["documents"]:
        return {"needs_retry": True}
    return {"needs_retry": False}
