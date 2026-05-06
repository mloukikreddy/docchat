import os
import shutil

print("Loading rag_pipeline...")

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from file_loader import load_file
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

SESSIONS_DIR = "sessions"

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)

_embeddings = None
_llm = None

print("Directories created")


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print("Loading embeddings...")

        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Embeddings loaded")

    return _embeddings


def get_llm():
    global _llm

    if _llm is None:
        print("Loading Groq model...")

        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        print("Groq model loaded")

    return _llm


PROMPT = """
You are a helpful AI assistant.

Use the provided context to answer the question accurately.
If the answer is not available in the context, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""


session_store = {}


def search_web(query: str, max_results: int = 3) -> str:
    try:
        print("Searching web...")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ""

        return "\n\n".join(
            f"[{r['title']}]\n{r['body']}"
            for r in results
        )

    except Exception as e:
        print(f"Web search failed: {e}")
        return ""


def ingest_file(session_id: str, file_path: str):

    print("Ingesting file...")

    embeddings = get_embeddings()

    text = load_file(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    docs = [Document(page_content=c) for c in chunks]

    session_index_path = os.path.join(SESSIONS_DIR, session_id)

    if os.path.exists(session_index_path):

        vectorstore = FAISS.load_local(
            session_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        vectorstore.add_documents(docs)

    else:

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

    vectorstore.save_local(session_index_path)

    session_store[session_id] = {
        "vectorstore": vectorstore,
        "history": [],
        "all_chunks": chunks
    }

    print("File indexed successfully")


def ask(session_id: str, question: str, use_web: bool = True) -> dict:

    print("Processing question...")

    embeddings = get_embeddings()

    doc_context = ""
    sources = []

    has_docs = (
        session_id in session_store or
        os.path.exists(os.path.join(SESSIONS_DIR, session_id))
    )

    if has_docs:

        if session_id not in session_store:

            vectorstore = FAISS.load_local(
                os.path.join(SESSIONS_DIR, session_id),
                embeddings,
                allow_dangerous_deserialization=True
            )

            session_store[session_id] = {
                "vectorstore": vectorstore,
                "history": [],
                "all_chunks": []
            }

        vectorstore = session_store[session_id]["vectorstore"]

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(question)

        doc_context = "\n\n".join(
            d.page_content for d in docs
        )

        sources = [d.page_content[:120] for d in docs]

    web_context = search_web(question) if use_web else ""

    context = ""

    if doc_context:
        context += f"=== DOCUMENT CONTEXT ===\n{doc_context}\n\n"

    if web_context:
        context += f"=== WEB CONTEXT ===\n{web_context}"

    if not context:
        context = "No additional context available."

    llm = get_llm()

    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["context", "question"]
    )

    chain = prompt | llm | StrOutputParser()

    final_answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("Answer generated")

    return {
        "answer": final_answer,
        "sources": sources[:3],
        "web_used": use_web and bool(web_context),
        "model_used": "Llama 3.3 via Groq"
    }


def clear_session(session_id: str):

    print(f"Clearing session: {session_id}")

    path = os.path.join(SESSIONS_DIR, session_id)

    if os.path.exists(path):
        shutil.rmtree(path)

    session_store.pop(session_id, None)

    print("Session cleared")