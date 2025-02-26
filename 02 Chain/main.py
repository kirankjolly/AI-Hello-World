from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import argparse

load_dotenv()
#RUN BELOW COMMAND IN THE TERMINAL FOR TESTING THE CODE
# python main.py --task "return a number pyramid" --language "c++"
parser =  argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()


code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="write a very short {language} code method that will {task}"
)

code_chain = code_prompt | llm

'''
Alternatively, you can use the LLMChain class to chain multiple prompts together:
from langchain.chains import LLMChain
code_chain = LLMChain(prompt=code_prompt, llm=llm)
'''

result = code_chain.invoke({
    "language": args.language,
    "task": args.task
})

print(result)
