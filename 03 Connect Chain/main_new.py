from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import argparse

load_dotenv()

# WITHOUT CHAINS
# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# ---------- STEP 1: Generate code ----------
code_messages = [
    SystemMessage(
        content="You are a senior software engineer. Return only code."
    ),
    HumanMessage(
        content=f"Write a very short {args.language} code method that will {args.task}."
    )
]

code_response = llm.invoke(code_messages)
code = code_response.content

# ---------- STEP 2: Generate test ----------
test_messages = [
    SystemMessage(
        content="You are a senior software engineer. Return only test code."
    ),
    HumanMessage(
        content=(
            f"Write a test for the following {args.language} code:\n\n{code}"
        )
    )
]

test_response = llm.invoke(test_messages)
test = test_response.content

# ---------- OUTPUT ----------
print("\n\n\nCODE GENERATED:")
print(code)
print("-----------------------")
print("\n\n\nTEST GENERATED:")
print(test)
