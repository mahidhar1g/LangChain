import streamlit as st
from models.embeddings import get_embedding_model
from models.openai_llm import get_llm
from tools import document_retrieval_tool, web_search_tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain import hub

load_dotenv()


embeddings = get_embedding_model()
llm = get_llm()


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def main():
    st.title("Conversational Agent with Streamlit")
    st.write("Interact with the AI-powered chatbot enhanced with document retrieval and web search capabilities.")

    prompt = hub.pull("hwchase17/react")
    tools = [document_retrieval_tool, web_search_tool]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question:")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        response = agent_executor.invoke({"input": user_input})
        with st.chat_message("assistant"):
            st.markdown(response["output"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})

if __name__ == "__main__":
    main()
