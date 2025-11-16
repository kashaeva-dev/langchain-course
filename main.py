from environs import Env
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI

env = Env()
env.read_env()

tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
llm_openai = ChatOpenAI(model="gpt-5-mini", temperature=1)
llm_deepseek = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_googlegenai = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = llm_googlegenai

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="generate"
    )

chain = agent_executor

def main():
    print("Hello from langchain-course!")
    result = chain.invoke({"input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"})
    print(result)

if __name__ == "__main__":
    main()
