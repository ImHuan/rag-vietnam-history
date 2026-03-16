from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient  
from modules import config
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

embeddings = HuggingFaceEndpointEmbeddings(
    model=config.EMBEDDING_MODEL, 
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

db = Qdrant(
    client=client,   
    collection_name=config.COLLECTION_NAME,
    embeddings=embeddings
)

def get_relevant_chunks(query: str, top_k: int = 4):
    results = db.similarity_search(query, k=top_k)
    return results
