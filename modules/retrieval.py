from langchain_community.vectorstores import Qdrant
from modules import config
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

embeddings = HuggingFaceEndpointEmbeddings(
    model=config.EMBEDDING_MODEL, 
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# connect
db = Qdrant(
    url= QDRANT_URL,
    api_key= QDRANT_API_KEY,
    collection_name=config.COLLECTION_NAME,
    embeddings=embeddings
)

def get_relevant_chunks(query: str, top_k: int = 4):
    results = db.similarity_search(query, k=top_k)
    return results
