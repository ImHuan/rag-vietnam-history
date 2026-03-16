FILE_PDF = "data/Dai-viet-su-ki-toan-thu.pdf"

COLLECTION_NAME = "vietnam_history"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#Model
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
TEMPERATURE = 0.3
TOP_K = 4
#prompt
SYSTEM_PROMPT = """
Bạn là trợ lý lịch sử Việt Nam. 
Hãy trả lời câu hỏi dựa trên nội dung trích từ Đại Việt sử ký toàn thư.
Chỉ sử dụng thông tin trong context được cung cấp.
Nếu context không có thông tin, hãy trả lời "Không tìm thấy trong tài liệu".
"""