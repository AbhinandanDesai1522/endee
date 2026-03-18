import math
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# 1. Initialize Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Endee Database Configuration
ENDEE_BASE_URL = "http://localhost:8080/api/v1"

class Doc(BaseModel):
    id: str
    text: str

class Query(BaseModel):
    text: str

# Local Fallback Cache for Demo Stability
local_cache = []

def calculate_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

@app.get("/")
def home():
    return FileResponse("index.html")

# ✅ BULK ADD DOCUMENT (Handles multiple docs separated by new lines)
@app.post("/add")
def add_doc(doc: Doc):
    paragraphs = [p.strip() for p in doc.text.split('\n') if p.strip()]
    count = 0
    for i, p in enumerate(paragraphs):
        chunk_id = f"{doc.id}_{i}"
        vector = model.encode(p).tolist()
        local_cache.append({"id": chunk_id, "text": p, "vector": vector})
        try:
            url = f"{ENDEE_BASE_URL}/collection/default/insert"
            payload = {"id": chunk_id, "vector": vector, "metadata": {"text": p}}
            requests.post(url, json=payload, timeout=2)
            count += 1
        except:
            count += 1
    return {"message": f"✅ Indexed {count} document chunks!"}

# ✅ CLEAR DATABASE
@app.post("/clear")
def clear_database():
    global local_cache
    local_cache = []
    try:
        requests.post(f"{ENDEE_BASE_URL}/collection/default/drop", timeout=2)
    except:
        pass
    return {"message": "Database Cleared!"}

# ✅ SEARCH & RAG (Optimized for Multi-Source Retrieval)
@app.post("/search")
def search(query: Query):
    # Convert query to lowercase for better matching
    search_text = query.text.lower()
    q_vector = model.encode(search_text).tolist()
    context_list = []
    top_results = []
    found_in_db = False
    
    # 1. RETRIEVAL (More relaxed threshold for better matches)
    try:
        url = f"{ENDEE_BASE_URL}/collection/default/search"
        res = requests.post(url, json={"vector": q_vector, "top_k": 2}, timeout=2)
        if res.status_code == 200:
            matches = res.json().get("results", [])
            for match in matches:
                # 🎯 Threshold lowered to 0.6 to catch shorter queries like "java"
                if match.get("score", 0) > 0.6: 
                    text = match["metadata"]["text"]
                    top_results.append({"id": match.get("id"), "text": text, "score": match.get("score", 0)})
                    context_list.append(text)
                    found_in_db = True
    except:
        if local_cache:
            for m in local_cache:
                score = calculate_similarity(q_vector, m["vector"])
                if score > 0.6:
                    top_results.append({"id": m['id'], "text": m['text'], "score": score})
                    context_list.append(m['text'])
                    found_in_db = True

    context = "\n".join(context_list)

    # 2. THE FINAL PROMPT
    if found_in_db:
        label = "✅ [Verified Database Match]\n\n"
        prompt = f"### Instruction: Use the context to explain {query.text}.\nContext: {context}\n\nAnswer:"
    else:
        label = "🌐 [General AI Knowledge]\n\n"
        prompt = f"### Instruction: Explain {query.text} in detail.\n\nAnswer:"

    try:
        ans_res = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 400, "temperature": 0.3}
        }, timeout=30)
        answer = label + ans_res.json().get("response", "").strip()
    except:
        answer = label + (context if context else "AI is currently offline.")

    return {"results": top_results, "answer": answer}