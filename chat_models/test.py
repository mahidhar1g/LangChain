from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatOpenAI(model='gpt-4o')

chat_history = []

while True:
    user_query = input("User: ")
    if user_query.lower() == "exit":
        break
    chat_history.append(HumanMessage(user_query))
    
    result = llm_model.invoke(chat_history)
    ai_response = result.content
    
    chat_history.append(AIMessage(ai_response))
    
    print(f"AI: {ai_response}")