from environs import Env
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

env = Env()
env.read_env()

# Сначала определим примеры "ввод-вывод"
examples = [
    {"input": "дом", "output": "🏠"},
    {"input": "компьютер", "output": "💻"},
    {"input": "погода", "output": "🌤️"},
]

# Создадим шаблон для форматирования каждого примера
example_template = """
Слово: {input}
Эмодзи: {output}
"""
example_prompt = PromptTemplate(
    input_variables=["input", "output"], template=example_template
)

# Теперь создадим основной Few-Shot промпт
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,  # шаблон для примеров
    prefix="Переведи следующие слова в эмодзи. Используй предоставленные примеры:",
    suffix="Слово: {input}\nЭмодзи:",
    input_variables=["input"],
    example_separator="\n",  # Разделитель между примерами
)


llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
chain = few_shot_prompt | llm
response = chain.invoke({"input": "Nginx"})

print(response.content)
