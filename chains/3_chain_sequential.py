from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows about {animal}."),
        ("human", "Tell me {fact_count} facts")
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Transulation the following text to {language}: {text}")
    ]
)

prepare_for_animal_facts = RunnableLambda(lambda x: {"animal": "cat", "fact_count": "2"})

prepare_for_translation =  RunnableLambda(lambda x: {"text": x, "language": "french"})

chain = prepare_for_animal_facts | animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

print(chain.invoke(chain))