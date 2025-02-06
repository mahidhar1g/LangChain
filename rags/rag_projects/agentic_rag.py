import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, tool
from dotenv import load_dotenv
from langchain import hub
import os
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

# Embedding model
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Chat model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


def load_pdf(folder_path):
    """
    Loads all PDF files from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        list: A list of documents loaded from all PDFs.
    """
    
    documents = []
    full_text = ""
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                for doc in documents:
                    full_text += doc.page_content
        return documents, full_text
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF file: {e}")



def chunk_text(documents, full_text):
    """
    Split the text into chunks and assign metadata to each chunk.
    
    Args:
        documents (list): A list of documents.
        full_text (str): The full text to be split into chunks.
    
    Returns:
        list: A list of chunks with metadata.
    """
    
    # Split the text into chunks and assign metadata to each chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    raw_chunks = text_splitter.split_text(full_text)

    # Assign metadata to chunks
    chunks = []
    for doc in documents:
        current_page_text = doc.page_content
        current_page_metadata = doc.metadata
        for chunk in raw_chunks:
            if chunk in current_page_text or chunk[:50] in current_page_text:
                if not chunk in current_page_text:
                    chunks.append(
                        {
                            "text": chunk.strip(),
                            "metadata": {
                                "source":current_page_metadata["source"],
                                "page":current_page_metadata["page_label"] + "," + str(int(current_page_metadata["page_label"]) + 1)
                            }
                        }
                    )
                    break
                chunks.append(
                        {
                            "text": chunk.strip(),
                            "metadata": {
                                "source":current_page_metadata["source"],
                                "page":current_page_metadata["page_label"]
                            }
                        }
                    )
    return chunks


def create_pinecone_index():
    """
    Create a Pinecone index if it does not already exist.
    
    Returns:
        Index: A Pinecone index object.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "agentic-rag-pinecone"

    if index_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{index_name}' created.")
        except PineconeApiException as e:
            if e.status == 409:  # Conflict error, index already exists
                print(f"Index '{index_name}' already exists.")
            else: 
                raise e  
    else:
        print(f"Index '{index_name}' already exists.")

    index = pc.Index(index_name)
    return index

def upsert_data_to_pinecone(documents, full_text, index):
    """
    Upsert the text chunks and their embeddings into the Pinecone index.
    
    Args:
        documents (list): A list of documents.
        full_text (str): The full text to be split into chunks.
        index (Index): A Pinecone index object.
    """

    try:
        chunks = chunk_text(documents, full_text)
    except Exception as e:
        raise RuntimeError(f"An error occurred while splitting the text: {e}")
    
    # Check if data has already been upserted with the same chunk size
    upserted_flag_id = "upserted_flag"
    response = index.fetch(ids=[upserted_flag_id])
    if response and upserted_flag_id in response["vectors"]:
        stored_chunk_size = response["vectors"][upserted_flag_id]["metadata"].get("chunk_size")
        if stored_chunk_size == CHUNK_SIZE:
            print("Data already upserted with the same chunk size. Skipping upsert.")
        else:
            print("Chunk size has changed. Deleting existing vectors and upserting new ones.")
            index.delete(delete_all=True)
    else:
        print("No existing data found. Proceeding with upsert.")

    # Upsert embeddings into the Pinecone index
    if not response or upserted_flag_id not in response["vectors"] or stored_chunk_size != CHUNK_SIZE:
        for i, chunk in enumerate(chunks):
            chunk_embedding = embeddings.embed_query(chunk["text"])
            index.upsert([(str(i), chunk_embedding, {"text": chunk["text"], "metadata": str(chunk["metadata"])})])

        # Upsert a flag to indicate data has been upserted, including the chunk size
        index.upsert([(upserted_flag_id, [0.1] * 1536, {"text": "upserted_flag", "chunk_size": CHUNK_SIZE})])
        print("Finished upserting embeddings.")


# Document Retrieval Tool
@tool
def document_retrieval_tool(query):
    """
    Retrieve information from the embedded documents based on the query.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(current_dir, "documents")
    documents, full_text = load_pdf(documents_dir)
    try:
        index = create_pinecone_index()
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the Pinecone index: {e}")
    
    upsert_data_to_pinecone(documents, full_text, index)
    query_embedding = embeddings.embed_query(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    matched_data = [(match.metadata["text"], match.metadata) for match in response.matches]
    if matched_data:
        return "\n\n".join([f"[Page {page}] {text}" for text, page in matched_data])
    else:
        return "No relevant information found in the provided documents."

# Web Search Tool
@tool
def web_search_tool(query):
    """
    Perform a web search using DuckDuckGo and return relevant information.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)
        
def main():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """ You run in a loop of Thought, Action, PAUSE, Action_Response.
                            At the end of the loop you output an Answer.
                            
                            Use Thought to understand the question you have been asked.
                            Use Action to run one of the actions available to you - then return PAUSE.
                            Action_Response will be the result of running those actions.
                            
                            Your goal is to output an Answer that is helpful to the user. These are the actions you can take:
                            
                            You are a helpful assistant that can provide information from documents and the web. Whatever the question is first try to find the answer in the documents. If you can't find the answer in the documents, then try to find the answer on the web using DuckDuckGo. However, if you find the answer in the documents, you should provide the answer from the documents only. If you can't find the answer in the documents or on the web, you should say that you couldn't find the answer.
                            
                            At the end of the answer you provide, include the source of the information. 
                            - If you found the answer in a document, include the **page number** and **source**.
                            - If you found the answer on the web, include the **website link**.
                            
                            Available tools: {tools}
                            Available tool names: {tool_names}
                        """
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    tools = [document_retrieval_tool, web_search_tool]
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    while True:
        query = input("Ask a question (or type exit to quit): ")
        if query.lower() == "exit":
            break
        
        response = agent_executor.invoke({"input": query})
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()