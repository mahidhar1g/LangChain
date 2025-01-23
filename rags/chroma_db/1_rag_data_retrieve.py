import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Setting up file paths for the persistent Chroma vector store directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Loads the Chroma vector store from the persistent directory.
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Where does Gandalf meet Frodo?"

# Configures a retriever with similarity-based search parameters (e.g., top 10 results, minimum score threshold).
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.1
    },
)
relevent_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevent_docs, 1):
    print(f"Document {i}: {doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")