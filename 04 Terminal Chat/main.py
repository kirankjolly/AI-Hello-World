from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
# memory =    ConversationBufferMemory(
#      chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables = ["content", "messages"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")
    print(content)
    result = chain({"content": content})
    print(result["text"])