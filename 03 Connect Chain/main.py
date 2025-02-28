from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser =  argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()


code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="write a very short {language} code method that will {task}"
)

code_test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="write a test for the following {language} code:\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)
code_test_chain = LLMChain(
    llm=llm,
    prompt=code_test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains = [code_chain, code_test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

result = chain.invoke({
    "language": args.language,
    "task": args.task
})

print("\n\n\nCODE GENERATED:")
print(result["code"])
print("-----------------------")
print("\n\n\nTEST GENERATED:")
print(result["test"])