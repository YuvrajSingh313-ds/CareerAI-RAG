# 🚀 CareerAI — RAG-Powered Career Intelligence

> AI-powered Resume & Job Description analysis using Retrieval-Augmented Generation (RAG).

CareerAI helps students and job seekers understand how well their Resume matches a specific Job Description. It retrieves relevant information from the uploaded documents and uses an LLM to generate contextual career insights.

## ✨ Features

- 📄 Resume & JD upload
- 🔍 Semantic search
- 📊 Resume–JD match analysis
- 🎯 Skill-gap identification
- 💡 Project & improvement suggestions
- 💬 Context-aware career Q&A
- 🤖 RAG-powered LLM responses

## 🧠 Core AI Concepts

**RAG Pipeline**

```text
Resume + JD
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
LLM
    ↓
Career Insights
The system combines document retrieval, text embeddings, vector search, and LLM-based generation to produce responses grounded in the user's Resume and Job Description.

🛠️ Tech Stack

Python • Streamlit • FastAPI • Gemini API • Sentence Transformers • ChromaDB • RAG

📁 Structure
CareerAI-RAG/
├── app.py
├── api.py
├── database.py
├── requirements.txt
├── .env.example
├── notebooks/
└── README.md
⚙️ Run Locally
git clone https://github.com/YuvrajSingh313-ds/CareerAI-RAG.git
cd CareerAI-RAG

python -m venv rag_career_env
rag_career_env\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
Configure your .env using .env.example before running the application.

📌 Status

Working Prototype | RAG Pipeline Implemented | Deployment Ready

👨‍💻 Author

Yuvraj Singh
B.Tech CSE — Data Science / AI & DS
