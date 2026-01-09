from workflow import ask
import uuid
import os

# MAGIC #13: Session Persistence Across Restarts
# ===============================================
# Problem: Python variables (like thread_id) are lost when the app restarts
# Solution: Save the active thread_id to a file
#
# This file stores ONLY the current thread_id (e.g., "user_session" or "abc-123-uuid")
# The actual conversation data is in data/checkpoints.db
SESSION_FILE = "data/.current_session"

def load_session():
    """
    Load the last active thread_id from file.

    Why this is needed:
    -------------------
    When you type 'new', thread_id changes to a UUID.
    If you restart the app, we want to continue with that UUID, not reset to "user_session".

    Example:
    --------
    # Session 1: User chats with "user_session"
    data/.current_session contains: "user_session"

    # User types 'new'
    data/.current_session now contains: "abc-123-uuid"

    # App restarts
    load_session() reads file → "abc-123-uuid"
    User continues the UUID conversation, not the user_session one!
    """
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            return f.read().strip()
    # Default to user_session if no saved session
    return "user_session"

def save_session(thread_id):
    """
    Save the current thread_id to file.

    This is called when:
    1. User types 'new' - saves the new UUID
    2. (Could be called on every message to track the active conversation)
    """
    os.makedirs("data", exist_ok=True)
    with open(SESSION_FILE, 'w') as f:
        f.write(thread_id)

if __name__ == "__main__":
    # MAGIC #14: Automatic Session Resumption
    # ========================================
    # Load the last active thread_id from the file
    # This ensures continuity across app restarts
    thread_id = load_session()
    print(f"Resuming conversation: {thread_id}")
    print("Chat started. Type 'exit' to quit or 'new' to start a new conversation.\n")

    while True:
        q = input("Ask: ")
        if q == "exit":
            break
        elif q == "new":
            # Generate new session ID for new conversation
            thread_id = str(uuid.uuid4())
            save_session(thread_id)  # IMPORTANT: Save to file so it persists after restart!
            print(f"Started new conversation: {thread_id}\n")
            continue

        # MAGIC #15: The Complete Memory Flow
        # ===================================
        # When you call ask(q, thread_id):
        #
        # 1. workflow.py receives thread_id
        # 2. Creates config = {"configurable": {"thread_id": thread_id}}
        # 3. Calls app.invoke(..., config)
        # 4. LangGraph's SqliteSaver:
        #    a. Looks up thread_id in data/checkpoints.db
        #    b. Loads latest checkpoint (including all messages)
        #    c. Deserializes BLOB data back to Python objects
        # 5. Merges new input with loaded state (using add_messages)
        # 6. Executes graph nodes (retrieve → evaluate → generate)
        # 7. Each node updates state
        # 8. After each node, SqliteSaver:
        #    a. Serializes updated state to BLOB
        #    b. Saves new checkpoint to database with thread_id
        #    c. Links it to previous checkpoint (parent_checkpoint_id)
        # 9. Returns final answer
        # 10. Full conversation history now in database!
        #
        # Next time you use the same thread_id:
        # - All previous messages are automatically loaded
        # - Conversation continues seamlessly!
        print("\nAnswer:\n", ask(q, thread_id))
