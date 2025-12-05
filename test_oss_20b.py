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

    url: str = Field(description="The URL of the job posting")


class AgentResponse(BaseModel):
    """Schema for the agent's response."""

    answer: str = Field(
        description="The agent's detailed answer about the job postings. Do not use tables in the answer."
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of URLs for all job postings mentioned. Extract ALL URLs from search results.",
    )


llm_openai = ChatOpenAI(model="gpt-5.1", temperature=0)
llm_googlegenai = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_local = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    max_retries=3,
)

llm = llm_local
tools = [TavilySearch(include_domains=["linkedin.com"])]
agent = create_agent(
    model=llm, tools=tools, response_format=ToolStrategy(AgentResponse)
)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job postings for an AI engineer using langchain in the Bay Area on LinkedIn. List their details and extract all URLs."
            )
        }
    )
    structured_output = result.get("structured_response")

    if structured_output:
        print("\n=== STRUCTURED OUTPUT ===")
        print(f"Type: {type(structured_output)}")
        print(f"\nAnswer:\n{structured_output.answer}")
        print(f"\nSources ({len(structured_output.sources)}):")
        for i, source in enumerate(structured_output.sources, 1):
            print(f"{i}. {source.url}")
    else:
        print("\nNo structured_response found in result!")
        print(f"Result keys: {result.keys()}")
        print(f"Full result:\n{result}")


if __name__ == "__main__":
    main()
