from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import tool
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
import datetime


load_dotenv()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

llm = ChatOpenAI(model="gpt-4")

query = "What is the current time in London (You are in USA, eastern time zone)? Show the time in tehe london, I don't want date."

prompt_template = hub.pull("hwchase17/react")

tools = [get_system_time]

agent = create_react_agent(llm, tools, prompt_template)

agen_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print(agen_executor.invoke({"input": query}))