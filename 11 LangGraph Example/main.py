from workflow import ask
import uuid
import os

SESSION_FILE = "data/.current_session"

def load_session():
    """Load the last active thread_id from file"""
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            return f.read().strip()
    # Default to user_session if no saved session
    return "user_session"

def save_session(thread_id):
    """Save the current thread_id to file"""
    os.makedirs("data", exist_ok=True)
    with open(SESSION_FILE, 'w') as f:
        f.write(thread_id)

if __name__ == "__main__":
    # Load the last active session or start with user_session
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
            save_session(thread_id)
            print(f"Started new conversation: {thread_id}\n")
            continue

        print("\nAnswer:\n", ask(q, thread_id))
