"""
Utility script to view conversations stored in the database
"""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import json

def view_conversations():
    """View all conversations in readable format"""

    # Connect to the database
    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    # Get all unique thread_ids
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    threads = cursor.fetchall()

    print(f"\n{'='*80}")
    print(f"Found {len(threads)} conversation(s)")
    print(f"{'='*80}\n")

    for (thread_id,) in threads:
        print(f"\n{'='*80}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*80}")

        # Get the latest checkpoint for this thread
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Get the checkpoint data
            checkpoint_tuple = memory.get_tuple(config)

            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                state = checkpoint_tuple.checkpoint.get('channel_values', {})
                messages = state.get('messages', [])

                print(f"\nTotal messages: {len(messages)}")
                print(f"{'-'*80}")

                for i, msg in enumerate(messages, 1):
                    msg_type = msg.__class__.__name__
                    content = msg.content

                    if msg_type == "HumanMessage":
                        print(f"\n[{i}] USER:")
                        print(f"    {content}")
                    elif msg_type == "AIMessage":
                        print(f"\n[{i}] AI:")
                        print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
                    else:
                        print(f"\n[{i}] {msg_type}:")
                        print(f"    {content[:100]}{'...' if len(content) > 100 else ''}")

                # Show other state info
                if 'question' in state:
                    print(f"\n{'-'*80}")
                    print(f"Last Question: {state['question']}")
                if 'answer' in state:
                    print(f"Last Answer: {state['answer'][:100]}{'...' if len(state.get('answer', '')) > 100 else ''}")

            else:
                print("No checkpoint data found")

        except Exception as e:
            print(f"Error reading checkpoint: {e}")

    print(f"\n{'='*80}\n")
    conn.close()

def view_thread(thread_id):
    """View a specific conversation thread"""
    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        checkpoint_tuple = memory.get_tuple(config)

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            state = checkpoint_tuple.checkpoint.get('channel_values', {})
            messages = state.get('messages', [])

            print(f"\n{'='*80}")
            print(f"Conversation: {thread_id}")
            print(f"{'='*80}\n")

            for msg in messages:
                msg_type = msg.__class__.__name__
                content = msg.content

                if msg_type == "HumanMessage":
                    print(f"USER: {content}")
                elif msg_type == "AIMessage":
                    print(f"AI: {content}")
                    print()
        else:
            print(f"No conversation found for thread_id: {thread_id}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # View specific thread
        thread_id = sys.argv[1]
        view_thread(thread_id)
    else:
        # View all conversations
        view_conversations()
