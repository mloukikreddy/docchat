# DocChat — Chat with Your Documents 📄

A RAG-powered document Q&A app built with FastAPI, LangChain, and multi-LLM fusion.

🔗 **Live Demo**: https://huggingface.co/spaces/loukikreddy22/docchat

## Features
- Upload PDF, DOCX, PPTX, TXT files
- Multi-LLM answers: Llama 3.3 + Gemini 2.0 + Cohere Command-R
- Web search augmentation via DuckDuckGo
- FAISS vector store for semantic retrieval
- Session-based multi-file support

## Tech Stack
- **Backend**: FastAPI, LangChain, FAISS
- **LLMs**: Groq (Llama 3.3), Google Gemini 2.0, Cohere Command-R
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Frontend**: Vanilla HTML/CSS/JS
- **Deployment**: HuggingFace Spaces (Docker)

## Run Locally
```bash
git clone https://github.com/mloukikreddy/docchat
cd docchat
pip install -r requirements.txt
uvicorn main:app --reload
```
