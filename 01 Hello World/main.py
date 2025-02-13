from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
llm = OpenAI()


result = llm.invoke("Write a very very short poem")

print(result)
