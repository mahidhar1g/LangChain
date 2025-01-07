from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows about {animal}."),
        ("human", "Tell me {fact_count} facts")
    ]
)

# Create individual runnables (steps in the chain)

# Just replaces the placeholder values
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# Converts the data into write structure for the model
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

# To extract the content from the response
parse_output = RunnableLambda(lambda x: x.content)


# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle= [invoke_model], last=parse_output)

response = chain.invoke({"animal": "dog", "fact_count": 2})

print(response)