from environs import Env
from langchain_openai import ChatOpenAI

env = Env()
env.read_env()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

result = llm.invoke("Скажи мне, кто такой Платон?")
print(result)
