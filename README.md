# RAG Document Search

A simple RAG web app that uploads PDF/DOCX/TXT/MD documents, stores metadata/chunks/embeddings in MongoDB, supports tags, and answers questions using OpenAI.

## Run locally

1. Start MongoDB (local MongoDB or MongoDB Atlas).
2. Copy `.env.example` to `.env` and set `MONGODB_URI` and `OPENAI_API_KEY`.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `uvicorn app.main:app --reload`
5. Open `http://localhost:8000`.

## Notes

This intentionally uses Python-side cosine similarity to keep the demo simple and MongoDB-compatible without requiring Atlas Vector Search. Do not commit `.env` or API keys.
