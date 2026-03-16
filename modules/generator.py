from dotenv import load_dotenv
import os
from groq import Groq
import config

# ==============================
# Load environment variables
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")


# =====================
# Create client
# =====================
client = Groq(api_key=GROQ_API_KEY)


# =====================
# Generator Function
# =====================
def generate_answer(query_text, documents):
    """
    query_text: user question
    documents: retrieved vector documents
    """

    # ===== Build context from retrieved chunks =====
    context_text = "\n\n".join([doc.page_content for doc in documents])

    # ===== Final Prompt =====
    final_prompt = f"""
{config.SYSTEM_PROMPT}

Context:
{context_text}

Question:
{query_text}
"""

    # ===== Call Groq API =====
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        temperature=config.TEMPERATURE,
    )

    return response.choices[0].message.content