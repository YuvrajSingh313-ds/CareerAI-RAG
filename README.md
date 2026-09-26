# 🚀 CareerAI — RAG-Powered Career Intelligence Platform

> An AI-powered career assistant that analyzes a Resume and Job Description (JD) to provide personalized job-fit insights, skill-gap analysis, and career recommendations.

CareerAI uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the user's Resume and Job Description before generating responses with an LLM.

---

## 🎯 Key Features

- 📄 Upload Resume in PDF, DOCX, TXT, or Markdown format
- 💼 Upload a specific Job Description
- 🔍 Semantic search over Resume and JD content
- 📊 Resume–JD match analysis
- 🎯 Identify missing or weak skills
- 💡 Suggest improvements and relevant projects
- 💬 Ask questions based on the uploaded Resume + JD
- 🤖 RAG-based LLM responses

---

## 🧠 AI / ML Concepts

### Retrieval-Augmented Generation (RAG)

CareerAI does not rely only on the LLM's general knowledge. It first retrieves relevant information from the uploaded documents and provides that context to the LLM.

```text
Resume + Job Description
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
   Relevant Context
          ↓
        LLM
          ↓
 Career Analysis
Semantic Search

The system uses text embeddings to represent document chunks as vectors. Semantic similarity is then used to retrieve the most relevant information for a user's question.

Vector Database

ChromaDB is used to store embeddings and efficiently retrieve relevant document content.

LLM

The retrieved context is combined with the user's question and passed to an LLM to generate a contextual career response.

🏗️ Architecture
              User
               │
       ┌───────┴────────┐
       │                │
    Resume             JD
       │                │
       └───────┬────────┘
               ↓
        Document Processing
               ↓
          Text Chunking
               ↓
        Embedding Model
               ↓
           ChromaDB
               ↓
       Semantic Retrieval
               ↓
        Retrieved Context
               ↓
          LLM + RAG
               ↓
       CareerAI Response
🛠️ Tech Stack
Technology	Purpose
Python	Core application development
Streamlit	Web interface
ChromaDB	Vector database
Embeddings	Semantic representation of text
Ollama / LLM	AI response generation
RAG	Context-aware AI generation
Git & GitHub	Version control
📁 Project Structure
CareerAI-RAG/
│
├── app.py              # Main Streamlit application
├── api.py              # AI/RAG application logic
├── database.py         # Database operations
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Ignored files and folders
│
├── notebooks/
│   └── career_ai.ipynb # Development / experimentation
│
└── README.md           # Project documentation
⚙️ Run Locally
1. Clone the repository
git clone https://github.com/YuvrajSingh313-ds/CareerAI-RAG.git
cd CareerAI-RAG
2. Create a virtual environment
python -m venv rag_career_env

Activate on Windows:

rag_career_env\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file using .env.example as the template.

Never commit API keys or other secrets to GitHub.

5. Run CareerAI
streamlit run app.py

The application will open in your browser.

🔄 How It Works
User uploads a Resume and Job Description.
CareerAI extracts and processes the document text.
Text is divided into meaningful chunks.
Embeddings are generated for the chunks.
Embeddings are stored in ChromaDB.
User asks a career-related question.
Relevant information is retrieved using semantic search.
Retrieved context is provided to the LLM.
The LLM generates a grounded response.
📌 Project Status

Working Prototype — Local Version Completed

Core RAG pipeline, document processing, semantic retrieval, ChromaDB integration, career analysis, and conversational Q&A are implemented.

🚧 Public deployment is the next step.

🔮 Future Improvements
Public cloud deployment
Faster LLM inference
Improved Resume–JD scoring
Skill-gap visualization
Multiple job comparison
Personalized learning roadmap
Improved mobile experience
👨‍💻 Author

Yuvraj Singh
B.Tech — Computer Science & Engineering
Specialization: Data Science / AI & DS
