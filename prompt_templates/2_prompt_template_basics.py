import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt = """
You are a helpful AI assistant.
User question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = prompt_template | llm | StrOutputParser()

print(chain.invoke({"question": "What is the capital of France?"}))