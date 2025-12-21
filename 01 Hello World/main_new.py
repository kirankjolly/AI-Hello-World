from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

response = llm.invoke([
    HumanMessage(content="Write a very very short poem")
])

print(response.content)
