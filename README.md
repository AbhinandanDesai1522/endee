# Endee AI Pro: Advanced Hybrid RAG System

A sophisticated Retrieval-Augmented Generation (RAG) application built for the **Endee.io** technical assessment.

## 🚀 Key Features
- **Hybrid RAG Logic**: Intelligently distinguishes between verified database context and general AI knowledge.
- **Vector Search Engine**: Integrated with the **Endee Vector Database** via Docker.
- **Local Inference**: Uses **Ollama (TinyLlama)** for private, secure generation.
- **Similarity Thresholding**: Prevents hallucinations by strictly filtering low-confidence matches.
- **Modern UI**: Animated, responsive dashboard built with **Tailwind CSS**.

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **Vector DB**: Endee Vector DB
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Frontend**: Glassmorphism UI (Tailwind/JS)

## 🏃 Setup
1. `docker compose up -d`
2. `ollama run tinyllama`
3. `uvicorn main:app --reload`
4. Visit `http://127.0.0.1:8000`

---
**Developed by:** Abhinandan Desai