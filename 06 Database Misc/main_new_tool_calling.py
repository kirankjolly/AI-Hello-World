import sqlite3
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

load_dotenv()

#THIS IS A TOOL CALLING EXAMPLE. BUT TOOLS ARE NOT MANDATORY

# ------------------ DB ------------------d
conn = sqlite3.connect("db.sqlite")

def run_sqlite_query(query: str):
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()

# ------------------ TOOL DEFINITION ------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_sqlite_query",
            "description": "Run a read-only SQL query on the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQLite SELECT query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ------------------ LLM ------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ------------------ STEP 1: Ask the model ------------------
messages = [
    SystemMessage(
        content=(
            "You are an assistant with access to a SQLite database.\n"
            "Only use SELECT queries.\n"
            "If you need data, call the tool."
        )
    ),
    HumanMessage(content="How many orders are there?")
]

response = llm.invoke(messages, tools=tools)

# ------------------ STEP 2: Did the model request a tool? ------------------
if response.tool_calls:
    tool_call = response.tool_calls[0]
    args = tool_call["args"]

    print("🔧 Tool called:", tool_call["name"])
    print("🧾 SQL:", args["query"])

    # Execute tool
    result = run_sqlite_query(args["query"])

    # ------------------ STEP 3: Send result back to model ------------------
    messages.append(response)
    messages.append(
        ToolMessage(
            tool_call_id=tool_call["id"],
            content=str(result)
        )
    )

    final_response = llm.invoke(messages)
    print("\n✅ Final Answer:\n")
    print(final_response.content)

else:
    print("❌ No tool was called")
