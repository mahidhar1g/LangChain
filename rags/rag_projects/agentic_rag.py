import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.conversation.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os
load_dotenv()


# Load the PDF file and extract the text
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
            if file_name.endswith("G.pdf"):
                file_path = os.path.join(folder_path, file_name)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                for doc in documents:
                    full_text += doc.page_content
        return documents, full_text
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF file: {e}")

current_dir = os.path.dirname(os.path.abspath(__file__))
documents_dir = os.path.join(current_dir, "documents")
documents, full_text = load_pdf(documents_dir)


# Split the text into chunks and assign metadata to each chunk
chunk_size = 500
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)
raw_chunks = text_splitter.split_text(full_text)

chunks = []
for i, doc in enumerate(documents):
    current_page_text = doc.page_content
    current_page_metadata = doc.metadata
    print(current_page_metadata)
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


print(len(chunks))
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} - {chunk}")
    print("")