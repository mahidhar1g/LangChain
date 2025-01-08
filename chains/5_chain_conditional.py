from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

postive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Assistant."),
        ("human", "Genrate a response for this postive feeback: {feedback}.")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Genrate a response addressing this neagtive feedback: {feedback}.")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Genrate a response addressing this neutral feedback: {feedback}.")
    ]
)

escalete_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Escalete this feedback to the manager: {feedback}.")
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a healpful assistant."),
        ("human", "Classify the sentiment of this feddback as postive, negative, neutral or escalte : {feedback}.")
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        postive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalete_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

print(chain.invoke({"feedback": "Teh product is ok."}))