# CareerAI — RAG-Powered Resume & Job Description Intelligence

CareerAI is an AI-powered career analysis application that compares a candidate's **Resume/CV** with a specific **Job Description (JD)** using **Retrieval-Augmented Generation (RAG)**.

The system extracts information from both documents, creates a searchable vector representation, retrieves relevant resume/JD information, and uses an LLM to generate a grounded career analysis.

## 🚀 Features

- Upload Resume/CV in PDF, DOCX, TXT or Markdown format
- Upload Job Description in PDF, DOCX, TXT or Markdown format
- Resume–JD compatibility / match score
- Identification of:
  - Strong matches
  - Missing or weak skills
  - Areas for improvement
  - Recommended projects
  - Career suggestions
- Ask follow-up questions about the uploaded Resume and JD
- RAG-based document retrieval
- Semantic search using vector embeddings
- LLM-generated career analysis
- Session-based conversation
- Local vector database using ChromaDB

## 🧠 How It Works

```text
Resume + Job Description
          ↓
   Document Extraction
          ↓
      Text Chunking
          ↓
   Embedding Generation
          ↓
     ChromaDB Vector Store
          ↓
   Semantic Retrieval
          ↓
      Relevant Context
          ↓
       LLM / RAG
          ↓
 Resume–JD Career Analysis
          ↓
 Match Score + Strengths + Gaps
 + Improvements + Project Suggestions
          ↓
       User Q&A