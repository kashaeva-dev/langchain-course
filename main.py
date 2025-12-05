from typing import List

from environs import Env
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

env = Env()
env.read_env()


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response."""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer",
    )


llm_openai = ChatOpenAI(model="gpt-5.1", temperature=0)
llm_googlegenai = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

llm = llm_googlegenai
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse)
    )


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print(result.get("structured_response"))


if __name__ == "__main__":
    main()
