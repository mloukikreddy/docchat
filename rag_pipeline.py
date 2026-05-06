import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from file_loader import load_file
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)

_embeddings = None
_llms = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def get_llms():
    global _llms
    if _llms is None:
        _llms = {
            "groq": ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
            "gemini": ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")),
            "cohere": ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"))
        }
    return _llms

INDIVIDUAL_PROMPT = """You are a helpful assistant. Answer the question using the context below.
Be detailed and accurate.

{context}

Question: {question}

Answer:"""

MERGE_PROMPT = """You are a master summarizer. Below are answers from 3 different AI systems.
Combine into one comprehensive, well-structured answer. Remove duplicates. Keep best insights.

Question: {question}

Answer from Llama:
{answer1}

Answer from Gemini:
{answer2}

Answer from Cohere:
{answer3}

Combined answer:"""

session_store = {}

def search_web(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n\n".join(f"[Web - {r['title']}]\n{r['body']}" for r in results)
    except Exception as e:
        print(f"Web search failed: {e}")
        return ""

def query_single_model(llm, question: str, context: str) -> str:
    try:
        prompt = PromptTemplate(
            template=INDIVIDUAL_PROMPT,
            input_variables=["context", "question"]
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    except Exception as e:
        return f"[Model unavailable: {str(e)[:100]}]"

def query_all_models(context: str, question: str) -> dict:
    llms = get_llms()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(query_single_model, llms["groq"], question, context)
        f2 = executor.submit(query_single_model, llms["gemini"], question, context)
        f3 = executor.submit(query_single_model, llms["cohere"], question, context)
        return {"llama": f1.result(), "gemini": f2.result(), "cohere": f3.result()}

def merge_answers(question: str, answers: dict) -> str:
    try:
        llms = get_llms()
        prompt = PromptTemplate(
            template=MERGE_PROMPT,
            input_variables=["question", "answer1", "answer2", "answer3"]
        )
        chain = prompt | llms["groq"] | StrOutputParser()
        return chain.invoke({
            "question": question,
            "answer1": answers["llama"],
            "answer2": answers["gemini"],
            "answer3": answers["cohere"]
        })
    except Exception:
        return answers["llama"]

def ingest_file(session_id: str, file_path: str):
    embeddings = get_embeddings()
    text = load_file(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    session_index_path = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(session_index_path):
        vectorstore = FAISS.load_local(
            session_index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(session_index_path)
    session_store[session_id] = {
        "vectorstore": vectorstore,
        "history": session_store.get(session_id, {}).get("history", []),
        "all_chunks": chunks
    }

def ask(session_id: str, question: str, use_web: bool = True) -> dict:
    embeddings = get_embeddings()
    doc_context = ""
    sources = []

    has_docs = session_id in session_store or os.path.exists(
        os.path.join(SESSIONS_DIR, session_id)
    )

    if has_docs:
        if session_id not in session_store:
            vectorstore = FAISS.load_local(
                os.path.join(SESSIONS_DIR, session_id), embeddings,
                allow_dangerous_deserialization=True
            )
            session_store[session_id] = {"vectorstore": vectorstore, "history": [], "all_chunks": []}

        vectorstore = session_store[session_id]["vectorstore"]
        all_chunks = session_store[session_id].get("all_chunks", [])

        summary_keywords = ["summary", "summarize", "overview", "what is this", "about", "explain", "describe"]
        is_summary = any(kw in question.lower() for kw in summary_keywords)

        if is_summary and all_chunks:
            doc_context = "\n\n".join(all_chunks[:10])
            sources = [c[:120] for c in all_chunks[:3]]
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
            doc_context = "\n\n".join(d.page_content for d in docs)
            sources = [d.page_content[:120] for d in docs]

    session_store.setdefault(session_id, {"history": [], "all_chunks": []})
    history = session_store[session_id]["history"]

    web_context = search_web(question) if use_web else ""

    context = ""
    if doc_context:
        context += f"=== FROM YOUR UPLOADED DOCUMENTS ===\n{doc_context}\n\n"
    if web_context:
        context += f"=== FROM WEB SEARCH ===\n{web_context}"
    if not context:
        context = "No document context. Answer from general knowledge."

    answers = query_all_models(context, question)
    final_answer = merge_answers(question, answers)
    history.append({"q": question, "a": final_answer})

    return {
        "answer": final_answer,
        "sources": sources[:3],
        "web_used": use_web and bool(web_context),
        "models_used": ["Llama 3.3", "Gemini 2.0", "Cohere Command-R"]
    }

def clear_session(session_id: str):
    path = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(path):
        shutil.rmtree(path)
    session_store.pop(session_id, None)