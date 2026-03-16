FILE_PDF = "data/The Diabetes Code.pdf"

COLLECTION_NAME = "diabetes-code"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#Model
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
TOP_K = 10
#prompt
SYSTEM_PROMPT = """
You are an expert assistant answering questions about the book "The Diabetes Code".

Your task is to answer the user's question using ONLY the information provided in the extracted text.

Rules:
- Base your answer strictly on the provided context.
- Do not invent information.
- If the answer cannot be found in the text, say:
  "The provided text does not contain enough information to answer this question."
- Provide clear, concise explanations about diabetes, insulin resistance, metabolism, diet, and fasting when relevant.
- Write the answer in clear academic English.
"""
