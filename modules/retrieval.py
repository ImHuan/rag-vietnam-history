from langchain_community.vectorstores import Qdrant
from modules import config
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    model=config.EMBEDDING_MODEL, 
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# connect
db = Qdrant(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
    collection_name=config.COLLECTION_NAME,
    embeddings=embeddings
)

def get_relevant_chunks(query: str, top_k: int = 4):
    results = db.similarity_search(query, k=top_k)
    return results
