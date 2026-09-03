import os
from datetime import datetime, timezone
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient
from pypdf import PdfReader
from docx import Document as DocxDocument

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "ragdocsearch")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

mongo = MongoClient(MONGODB_URI)
db = mongo[MONGODB_DB]
documents = db.documents
chunks = db.chunks
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="RAG Document Search")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class QueryRequest(BaseModel):
    question: str
    tags: list[str] = []
    top_k: int = 5


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = DocxDocument(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(400, "Supported files: PDF, DOCX, TXT, MD")


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    result = []
    start = 0
    while start < len(text):
        result.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return result


def embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@app.get("/")
def index():
    return FileResponse("app/static/index.html")

@app.get("/api/documents")
def list_documents():
    return [{"id": str(d["_id"]), "filename": d["filename"], "tags": d["tags"], "chunk_count": d["chunk_count"], "uploaded_at": d["uploaded_at"].isoformat()} for d in documents.find().sort("uploaded_at", -1)]

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), tags: str = Form("")):
    data = await file.read()
    text = extract_text(file.filename or "document", data)
    parts = split_text(text)
    if not parts:
        raise HTTPException(400, "The document contains no extractable text")
    tag_list = sorted({t.strip().lower() for t in tags.split(",") if t.strip()})
    vectors = embed(parts)
    now = datetime.now(timezone.utc)
    doc = {"filename": file.filename, "tags": tag_list, "uploaded_at": now, "content_type": file.content_type, "chunk_count": len(parts)}
    doc_id = documents.insert_one(doc).inserted_id
    chunks.insert_many([{"document_id": doc_id, "filename": file.filename, "tags": tag_list, "chunk_index": i, "text": part, "embedding": vector} for i, (part, vector) in enumerate(zip(parts, vectors))])
    return {"id": str(doc_id), "filename": file.filename, "tags": tag_list, "chunk_count": len(parts)}

@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is required")
    qvec = embed([req.question])[0]
    candidates = chunks.find({"tags": {"$all": req.tags}}) if req.tags else chunks.find({})
    scored = []
    for chunk in candidates:
        scored.append((cosine(qvec, chunk["embedding"]), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[: max(1, min(req.top_k, 10))]
    if not selected:
        return {"answer": "No indexed documents matched your query.", "sources": []}
    context = "\n\n".join(f"SOURCE: {c['filename']}\n{c['text']}" for _, c in selected)
    prompt = f"Answer the question using only the supplied context. If the answer is not in the context, say you don't know.\n\nCONTEXT:\n{context}\n\nQUESTION: {req.question}"
    response = openai_client.chat.completions.create(model=CHAT_MODEL, temperature=0, messages=[{"role": "system", "content": "You are a concise retrieval-augmented assistant."}, {"role": "user", "content": prompt}])
    sources = [{"filename": c["filename"], "chunk_index": c["chunk_index"], "score": round(score, 4)} for score, c in selected]
    return {"answer": response.choices[0].message.content, "sources": sources}

@app.get("/health")
def health():
    mongo.admin.command("ping")
    return {"status": "ok"}
