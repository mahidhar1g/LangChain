import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

if not os.path.exists(persistent_directory):
    print("Persistent directory doesn't exist. Creating it and initializing vector store...")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError("Book directory not found at path: " + books_dir)
    
    # Gives a list of all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    print(f"Book files: {book_files}")
    
    # Read the text content from each file and store it with metadata
    documtents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        
        for doc in book_docs:
            # Adds metadata to each documnent indicating its source
            doc.metadata = {"source": book_file}
            documtents.append(doc)
    
    # Splits the loaded text into smaller chunks using a character-based text splitter and for each chunk, the metadata also gets added.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documtents)
    
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
    
else:
    print("Persistent directory exists. Loading vector store...")
    