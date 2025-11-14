from environs import Env
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

env = Env()
env.read_env()



llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    print(result)


if __name__ == "__main__":
    main()
