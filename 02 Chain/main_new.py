from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import argparse

load_dotenv()

# WITHOUT CHAIN EXAMPLE
# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# LLM (chat-first)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Explicit prompt construction
messages = [
    SystemMessage(
        content="You are a senior software engineer. Return only code, no explanation."
    ),
    HumanMessage(
        content=f"Write a very short {args.language} code method that will {args.task}."
    )
]

response = llm.invoke(messages)

print(response.content)
