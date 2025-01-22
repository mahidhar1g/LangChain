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
    """Fetches data from the local Flask server."""
    url = "http://127.0.0.1:5000/"  # Flask server URL

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Failed to fetch data from Flask server. Status code: {response.status_code}"

llm = ChatOpenAI(model="gpt-4")

query = input("You: ")
prompt_template = hub.pull("hwchase17/react")
tools = [get_stock_data]
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print(agent_executor.invoke({"input": query}))
