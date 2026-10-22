from fastapi import FastAPI, UploadFile, Query
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF reading
import numpy as np

app = FastAPI()

# Load embedding model (offline HuggingFace)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Store Q&A chunks and embeddings
pdf_chunks = []
pdf_embeddings = None

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    """
    Upload a PDF file, split into Q&A pairs, and embed them.
    """
    global pdf_chunks, pdf_embeddings
    pdf_chunks = []

    # Read entire PDF text
    doc = fitz.open(stream=await file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc[page_num].get_text("text") + "\n"

    # Split by "Q" markers to capture full Q&A blocks
    qa_blocks = full_text.split("Q")
    for block in qa_blocks:
        block = block.strip()
        if block:
            text = "Q" + block
            # Extract the actual Q number
            try:
                q_num = int(text.split(":")[0][1:])
            except:
                q_num = None
            pdf_chunks.append({"page": None, "text": text, "q_num": q_num})

    # Embed all Q&A chunks
    texts = [chunk["text"] for chunk in pdf_chunks]
    pdf_embeddings = embed_texts(texts)

    return {"message": f"PDF uploaded with {len(pdf_chunks)} Q&A pairs."}

@app.get("/ask")
async def ask(query: str = Query(..., description="Enter your question or Q#")):
    """
    Ask either by question number (e.g. Q56) or by natural language query.
    """
    global pdf_chunks, pdf_embeddings
    if not pdf_chunks or pdf_embeddings is None:
        return {"error": "No PDF uploaded yet."}

    # If user asks by number, e.g. "Q56"
    if query.strip().upper().startswith("Q"):
        try:
            q_num = int(query.strip()[1:])
            for chunk in pdf_chunks:
                if chunk.get("q_num") == q_num:
                    return {
                        "question_number": q_num,
                        "answer_text": chunk["text"]
                    }
        except ValueError:
            pass  # fall back to embedding search

    # Otherwise, do embedding similarity search
    query_embedding = embed_texts([query])[0]
    similarities = np.dot(pdf_embeddings, query_embedding) / (
        np.linalg.norm(pdf_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    best_idx = int(np.argmax(similarities))
    best_chunk = pdf_chunks[best_idx]

    return {
        "question": query,
        "answer_text": best_chunk["text"]
    }
