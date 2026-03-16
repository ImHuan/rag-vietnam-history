from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from modules import config

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
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
