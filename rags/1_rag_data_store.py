
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Setting up file and directory paths for processing and storing vector data.
if not os.path.exists(persistent_directory):
    print("Persistent directory doens't exist. Creating it and initiliazing vector store...")
    
    if not os.path.exists(file_path):
        raise Exception("File not found at path: " + file_path)
    
    
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Splits the loaded text into smaller chunks using a character-based text splitter.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")
    
    
    print("\n--- Creating embessings ---")
    
    # Creates embeddings for the text chunks using OpenAI's embedding model.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished creating embedding ---")
    
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
    
else:
    print("Persistent directory exists. Loading vector store...")