from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
load_dotenv()

# Load the PDF file and extract the text
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "incorrect_facts.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist.")

text = ""
try:
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the PDF file: {e}")


# Split the text into chunks
chunk_size = 200
chunk_overlap = 10
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
chunks = text_splitter.split_text(text)


# Create a Pinecone index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-pinecone"

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

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)


# Check if data has already been upserted with the same chunk size
upserted_flag_id = "upserted_flag"
response = index.fetch(ids=[upserted_flag_id])
if response and upserted_flag_id in response["vectors"]:
    stored_chunk_size = response["vectors"][upserted_flag_id]["metadata"].get("chunk_size")
    if stored_chunk_size == chunk_size:
        print("Data already upserted with the same chunk size. Skipping upsert.")
    else:
        print("Chunk size has changed. Deleting existing vectors and upserting new ones.")
        index.delete(delete_all=True)
else:
    print("No existing data found. Proceeding with upsert.")


# Upsert embeddings into the Pinecone index
if not response or upserted_flag_id not in response["vectors"] or stored_chunk_size != chunk_size:
    for i, chunk in enumerate(chunks):
        chunk_embedding = embeddings.embed_query(chunk)
        index.upsert([(str(i), chunk_embedding, {"text": chunk})])
    
    # Upsert a flag to indicate data has been upserted, including the chunk size
    index.upsert([(upserted_flag_id, [0.1]*1536, {"text": "upserted_flag", "chunk_size": chunk_size})])
    print("Finished upserting embeddings.")


# Query the Pinecone index
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

while True:
    query = input("Ask a question (or type exit to quit): ")
    if query.lower() == "exit":
        break
    
    print(f"\nQuery: {query}")
    query_embedding = embeddings.embed_query(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    augmented_content = "\n\n".join([match.metadata["text"] for match in response.matches])
    print(f"\nAugmented content: {augmented_content}")
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant that strictly follows the provided context to answer questions. "
            "You **must not** use any external knowledge, even if the user asks you to. "
            "If the answer is not found in the provided context, respond with:\n"
            "'I can only answer based on the provided context, and no relevant information is available.'\n\n"
            "Context: {context}"),
            ("human", "Question: {question}")
        ]
    )

    
    prepare_prompt_template = RunnableLambda(lambda x: prompt_template.format_prompt(context=augmented_content, question=query))
    
    chain = prepare_prompt_template | llm | StrOutputParser()
    print(f"\nAnswer: {chain.invoke({})}")