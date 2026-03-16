from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import config
import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def create_vector_db():

    loader = PyMuPDFLoader(config.FILE_PDF)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    print("Đang đẩy dữ liệu lên Qdrant, vui lòng đợi...")
    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
        timeout=120,          
        force_recreate=True   
    )
    print("Khởi tạo Vector DB thành công!")


if __name__ == "__main__":
    create_vector_db()