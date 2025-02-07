import streamlit as st
from models.embeddings import get_embedding_model
from models.openai_llm import get_llm
from tools import document_retrieval_tool, web_search_tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
import os
from langchain.chains import RetrievalQA, LLMChain


load_dotenv()


# Embedding model
embeddings = get_embedding_model()

# Chat model
llm = get_llm()

        
def main():
    # prompt = hub.pull("hwchase17/react")
    # tools = [document_retrieval_tool, web_search_tool]
    # agent = create_react_agent(llm, tools, prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # while True:
    #     query = input("Ask a question (or type exit to quit): ")
    #     if query.lower() == "exit":
    #         break
        
    #     response = agent_executor.invoke({"input": query})
    #     print(f"\nAnswer: {response}")
    
    st.title("RAG Agent with Browser Search and Document Retrieval")
    st.write("Interact with the Agentic AI RAG chatbot powered by LangChain with added browser search and document retrieval features.")

    st.sidebar.title('Settings')
    system_prompt = st.sidebar.text_input("System prompt:", value="You are a helpful assistant.")
    model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma2-9b-it'])
    memory_length = st.sidebar.slider('Memory length:', 1, 10, value=5)




if __name__ == "__main__":
    main()
    