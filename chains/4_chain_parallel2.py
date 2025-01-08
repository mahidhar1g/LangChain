from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4o")

intial_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a math expert."),
        ("human", "What is the square root of {number1}.")
    ]
)

parallel1_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a math expert."),
        ("human", "What is {number2} times 2.")
    ]
)

parallel2_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a math expert."),
        ("human", "What is {number3} times 2.")
    ]
)

prompt_intial_template = RunnableLambda(lambda x: intial_template.format_prompt(number1=input("Enter a number: ")))
prompt_parallel1_template = RunnableLambda(lambda x:parallel1_template.format_prompt(number2=x))
prompt_parallel2_template = RunnableLambda(lambda x: parallel2_template.format_prompt(number3=x))

# chain = prompt_coffee_template | model | StrOutputParser()
chain = (
    prompt_intial_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={
        "parallel1": prompt_parallel1_template | model | StrOutputParser(),
        "parallel2": prompt_parallel2_template | model | StrOutputParser()
        })
    | RunnableLambda(lambda x: {
        f"{x['branches']['parallel1']} {x['branches']['parallel2']}"
        })
)

print(chain.invoke({}))