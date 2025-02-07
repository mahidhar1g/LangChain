# tools.py
import os
from langchain.agents import tool
from utils.pdf_utils import load_pdf
from utils.pinecone_utils import create_pinecone_index, upsert_data_to_pinecone
from langchain_community.tools import DuckDuckGoSearchRun
from models.embeddings import get_embedding_model


@tool
def document_retrieval_tool(query):
    """
    Retrieve information from the embedded documents based on the query.
    
    Additional Information on when to use this tool:
        - This should be the primary source of information.
        - If no relevant information is found, return an empty string or indicate that the data is unavailable.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(current_dir, "documents")
    documents, full_text = load_pdf(documents_dir)
    try:
        index = create_pinecone_index()
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the Pinecone index: {e}")
    
    upsert_data_to_pinecone(documents, full_text, index)
    
    embeddings = get_embedding_model()
    query_embedding = embeddings.embed_query(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    matched_data = [(match.metadata["text"], match.metadata) for match in response.matches]
    if matched_data:
        return "\n\n".join([f"[Page {page}] {text}" for text, page in matched_data])
    else:
        return "No relevant information found in the provided documents."


@tool
def web_search_tool(query):
    """
    Perform a web search using DuckDuckGo and return relevant information.
    
    When to use this tool:
        - If the document_retrieval_tool does not find relevant information.
        - Use this tool only when the needed information is missing from the document retrieval tool.
    """
    
    search = DuckDuckGoSearchRun()
    return search.run(query)