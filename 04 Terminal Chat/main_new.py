from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(
    model="gpt-4o-mini",   # or whatever model you use
    temperature=0.7
)

# ---- conversation state (replace with DB later) ----
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input(">> ")

    if user_input.lower() in ("exit", "quit"):
        break

    messages.append(HumanMessage(content=user_input))

    response = chat.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print(response.content)
