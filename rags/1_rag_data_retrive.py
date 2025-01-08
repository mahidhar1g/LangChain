import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.ansolute(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")