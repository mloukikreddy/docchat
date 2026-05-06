import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_pipeline import ingest_file, ask, clear_session

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="DocChat")
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED = {".pdf", ".txt", ".docx", ".pptx"}

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported: {ext}")
    if not session_id:
        session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(await file.read())
    ingest_file(session_id, save_path)
    return {"session_id": session_id, "filename": file.filename, "status": "indexed"}

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
def ask_question(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return ask(query.session_id, query.question)

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared"}