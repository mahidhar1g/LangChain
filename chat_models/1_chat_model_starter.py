from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke("Hello, how are you?")
print(result.content)