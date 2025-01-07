from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# This is how you can create a conversation with the model by passing a list of messages at the backend level.
messages = [
    SystemMessage("You are an expert in Trading and Finance."),
    HumanMessage("Give me a very short overview of the stock market."),
    AIMessage("The stock market is a platform where investors can buy and sell shares of publicly traded companies. It serves as a critical component of the global economy, facilitating capital formation and offering investors opportunities for profit. Key elements of the stock market include stock exchanges, such as the New York Stock Exchange (NYSE) and the Nasdaq, where trading occurs. Prices of stocks are influenced by factors like company performance, economic indicators, geopolitical events, and market sentiment. The stock market is often seen as a barometer of economic health and involves various participants, including individual investors, institutional investors, and market makers."),
    HumanMessage("How many stock exchanges are there in the world?"),
]

result = llm.invoke(messages)
print(result.content)