from datetime import date, datetime, time
from typing import List

from environs import Env
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

env = Env()
env.read_env()


class Action(BaseModel):
    name: str = Field(description="Имя животного")
    action: str = Field(description="Действие, которое было выполнено")
    action_date: date = Field(description="Дата, когда было выполнено действие")
    action_time: time = Field(description="Время, когда было выполнено действие")


parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template=(
        "Ты извлекаешь информацию о действии животного из текста.\n"
        "{format_instructions}\n\n"
        "Текст: {text}\n"
        "Текущее время: {now}\n\n"
        #       "Ответь ТОЛЬКО одним JSON-объектом, "
        #       "БЕЗ пояснений, текста до или после, без ```."
        #       "НЕ писать в ответе Действие животного: ..."
    ),
    input_variables=["text", "now"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


llm_openai = ChatOpenAI(model="gpt-5.1", temperature=0)
llm_ollama = ChatOllama(model="llama3.1:8b", temperature=0)
llm_deepseek = ChatDeepSeek(model="deepseek-reasoner", temperature=0)
llm_glm = ChatOpenAI(
    model="glm-4.6",
    base_url="https://api.z.ai/api/paas/v4/",
    api_key="5f49b10e2cd64edca8dc7da5b59a677f.tv56yZPsRS63lIuD",
    temperature=0,
)

llm = llm_openai

chain = prompt | llm | parser

result = chain.invoke(
    {"text": "Трикс покакал сейчас", "now": datetime.now().isoformat()}
)

print("Подождите, идёт обработка...")
print(result)
