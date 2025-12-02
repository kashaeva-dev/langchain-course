from environs import Env
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

env = Env()
env.read_env()

tools = [TavilySearch()]
# react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
format_instructions = output_parser.get_format_instructions()
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
    ).partial(format_instructions=output_parser.get_format_instructions())



llm_openai = ChatOpenAI(model="gpt-5-mini", temperature=1)
llm_deepseek = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_googlegenai = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_oss_20b = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)
llm = llm_oss_20b

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    )

chain = agent_executor

def main():
    print("Hello from langchain-course!")
    result = chain.invoke({"input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"})
    print(result)

if __name__ == "__main__":
    main()
