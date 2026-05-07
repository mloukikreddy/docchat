# DocChat — Chat with Your Documents 🤖📄

> Upload any document. Ask anything. Get answers powered by 3 AI models simultaneously.

🔗 **Live Demo**: https://huggingface.co/spaces/loukikreddy22/docchat

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace%20Spaces-yellow)

---

## 🚀 What is DocChat?

DocChat is a full-stack RAG (Retrieval-Augmented Generation) application that lets you upload documents and have intelligent conversations with them. It uses **3 LLMs simultaneously** and merges their answers for maximum accuracy.

---

## ✨ Features

- 📁 Upload PDF, DOCX, PPTX, TXT files
- 🧠 Multi-LLM fusion: Llama 3.3 + Gemini 2.0 Flash + Cohere Command-R
- 🌐 Web search augmentation via DuckDuckGo
- 🔍 Semantic search using FAISS vector store
- 💬 Session-based multi-file chat
- 🐳 Dockerized and deployed on HuggingFace Spaces

---

## 🏗️ System Architecture

```text
User Uploads Document/File
(PDF / DOCX / PPTX / TXT)
                │
                ▼
      File Parsing Layer
      (Text Extraction)
                │
                ▼
      Text Chunking
      (500 Tokens + 50 Token Overlap)
                │
                ▼
      Embedding Model
      (Vector Generation)
                │
                ▼
      FAISS Vector Store
      (Semantic Indexing)
                │
                ▼
      User Query Input
                │
                ▼
      Semantic Retrieval
      (Top 4 Relevant Chunks)
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
Context Retrieval    Web Search
From Documents       External Sources
        │                │
        └───────┬────────┘
                │
                ▼
      Multi-LLM Layer
      • Llama
      • Gemini
      • Cohere
      (Parallel Inference)
                │
                ▼
      Response Fusion
      (Llama Aggregates Final Answer)
                │
                ▼
      Final AI Response
      (Returned to User)
```

### ⚙️ Workflow Summary

1. User uploads a document (`PDF`, `DOCX`, `PPTX`, or `TXT`)  
2. The system extracts and preprocesses the document text  
3. Text is split into overlapping chunks for better context retention  
4. Embeddings are generated and stored in a `FAISS` vector database  
5. User submits a question  
6. Relevant chunks are retrieved using semantic similarity search  
7. Additional context is fetched from:
   - Uploaded documents  
   - External web sources  
8. Multiple LLMs (`Llama`, `Gemini`, and `Cohere`) generate responses simultaneously  
9. Llama combines and refines all generated outputs  
10. Final intelligent response is returned to the user  
    
---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLMs | Groq/Llama 3.3, Gemini 2.0 Flash, Cohere Command-R |
| Web Search | DuckDuckGo Search |
| Frontend | HTML, CSS, Vanilla JS |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |

---

## 🔧 Run Locally

```bash
# Clone repo
git clone https://github.com/mloukikreddy/docchat
cd docchat

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
COHERE_API_KEY=your_key

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📁 Project Structure

```bash
docchat/
├── main.py           # FastAPI app, routes
├── rag_pipeline.py   # RAG logic, LLMs, FAISS
├── file_loader.py    # PDF/DOCX/PPTX/TXT parser
├── static/
│   └── index.html    # Frontend UI
├── Dockerfile        # Container config
├── requirements.txt
└── .env              # API keys (not committed)

```
---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Get from console.groq.com |
| `GEMINI_API_KEY` | Get from aistudio.google.com |
| `COHERE_API_KEY` | Get from cohere.com |

---

## 👤 Author

**Mekala Loukik Reddy** — AI & Data Science Student, Anurag University  
[LinkedIn](https://linkedin.com/in/mekala-loukik-reddy-a717b4287) | [GitHub](https://github.com/mloukikreddy)
