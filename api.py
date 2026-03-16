from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modules.retrieval import get_relevant_chunks
from modules.generator import generate_answer
from fastapi.responses import FileResponse
import os

app = FastAPI()

@app.get("/")
def serve_frontend():
    if not os.path.exists("index.html"):
        return {"error": "index.html not found"}
    return FileResponse("index.html")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        question = request.question
        docs = get_relevant_chunks(question)
        answer = generate_answer(question, docs)
        return {"answer": answer}
    except Exception as e:
        import traceback
        print("=== BẮT ĐẦU BÁO LỖI ===")
        traceback.print_exc()  
        print("=== KẾT THÚC BÁO LỖI ===")
        raise HTTPException(status_code=500, detail=str(e))


