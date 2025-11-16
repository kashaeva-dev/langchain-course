from typing import List
from pydantic import BaseModel, Field

from environs import Env
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

env = Env()
env.read_env()


class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response."""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list ofsources used to generate the answer")
    

llm = ChatOpenAI(model="gpt-5.1", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(
            content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")}
    )
    print(result)


if __name__ == "__main__":
    main()
