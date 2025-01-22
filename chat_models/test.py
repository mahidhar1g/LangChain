from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import tool
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
import os
import requests

load_dotenv()

@tool
def get_stock_data():
    """Fetches AAPL stock data from Polygon API for the specified date range."""
    api_key = "YQK1ICVB1iiwDKOklC7Bl6Ea0f9PUPv5"
    if not api_key:
        return "Polygon API key not found. Please set it in the environment variables."

    url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-10-09/2024-12-10"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": api_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Failed to fetch stock data. Status code: {response.status_code}"

llm = ChatOpenAI(model="gpt-4")

query = input("You: ")
prompt_template = hub.pull("hwchase17/react")
tools = [get_stock_data]
agent = create_react_agent(llm, tools, prompt_template)
agen_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print(agen_executor.invoke({"input": query}))