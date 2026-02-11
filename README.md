# 🩺 Medical Chatbot using LangChain (RAG)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) based medical chatbot using LangChain, FAISS vector database, HuggingFace embeddings, and a Zephyr-7B LLM via Hugging Face Inference API.

The system retrieves relevant medical document chunks using semantic similarity search and generates grounded, context-aware responses through a Streamlit chat interface.

---

## Tech Stack
- LangChain
- FAISS (Vector Database)
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- HuggingFace LLM (`HuggingFaceH4/zephyr-7b-beta`)
- Streamlit
- Python

---

## Features
- Retrieval-Augmented Generation (RAG)
- Semantic similarity search using FAISS
- Context-grounded response generation
- Custom prompt for hallucination control
- Streamlit-based conversational UI
- Secure API token handling using `.env`
- Cached vectorstore loading for performance optimization

---

## Project Structure
```
medical-chatbot-main-Project/
│
├── medibot.py
├── requirements.txt
├── data/
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss
│       └── index.pkl
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root directory:

```
HF_TOKEN=your_huggingface_api_token
```

---

## Run the Application

```bash
streamlit run medibot.py
```

The application will launch in your browser.

---

## Architecture Flow

User Query  
⬇  
Embedding Generation  
⬇  
FAISS Similarity Search (Top-K Retrieval)  
⬇  
Context Retrieval  
⬇  
LLM Response Generation  
⬇  
Streamlit Output  

---

##  Skills Demonstrated
- RAG Pipeline Design
- Vector Database Integration
- Embedding-Based Semantic Search
- LLM API Integration
- Prompt Engineering
- Streamlit UI Development
- Secure Environment Variable Handling

---

## Author
Nitesh Godse
