from environs import Env
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

env = Env()
env.read_env()

template = "Ты - эксперт по JavaScript. Расскажи мне про {subject}."
prompt = PromptTemplate(
    template=template,
    input_variables=["subject"]
    )

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
chain = prompt | llm

print("Ждите, идёт обработка...")
result = chain.invoke({"subject": "функции"})
print(result)