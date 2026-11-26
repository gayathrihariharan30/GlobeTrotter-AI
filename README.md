# GlobeTrotter-AI

Project Overview
GlobeTrotter AI is a portfolio-ready, Retrieval-Augmented Generation (RAG) web application built in Python. It acts as an intelligent travel companion that ingests multiple destination PDF travel guides (such as Paris, Germany, and Amsterdam) and generates personalized, context-aware travel itineraries and recommendations.

Core Tech Stack & Architecture
Frontend / UI: Streamlit (web_app.py) providing an interactive web interface with customizable trip duration sliders, travel style selectors, and expandable source reference chunks.

LLM & Generation: Google Gemini (via langchain-google-genai), utilizing free-tier cloud models (gemini-3.8-flash / gemini-1.5-flash) to generate responses without requiring heavy local GPU storage.

Vector Database: ChromaDB, storing document chunks locally on disk (chroma_db) for fast semantic retrieval.

Embeddings & Document Processing: LangChain (PyPDFDirectoryLoader and RecursiveCharacterTextSplitter in ingest.py) paired with local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2) to ensure reliable local indexing without embedding API errors or heavy storage overhead.

Security & Portability: Implements dynamic credential handling through a password-protected Streamlit sidebar input (GOOGLE_API_KEY) rather than hardcoded secrets, ensuring safe public hosting on GitHub (gayathrihariharan30/GlobeTrotter-AI).

Key Technical Highlights
Hybrid RAG Pipeline: Combines local Hugging Face sentence-transformer embeddings with Google Gemini cloud inference to keep local hardware requirements minimal while maintaining fast retrieval.

Multi-Document Ingestion: Uses PyPDFDirectoryLoader to scan a docs/ directory and index multiple travel guides simultaneously into ChromaDB.

Production & Portfolio Readiness: Configured with .gitignore and requirements.txt files for clean version control and easy setup by recruiters or collaborators.
