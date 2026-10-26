# Legal-RAG-Chatbot
A Retrieval-Augmented Generation (RAG) based legal assistant for answering questions about the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023, using semantic search and LLMs.

# BNSS Legal Assistant – RAG Chatbot

An AI-powered legal assistant that retrieves relevant provisions from the **Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023** and generates context-grounded answers using Retrieval-Augmented Generation (RAG).

## Features

- Indexed 673 BNSS legal text chunks using ChromaDB.
- Semantic search using `BAAI/bge-base-en-v1.5`.
- Answer generation using Llama 3.2 (3B) through Ollama.
- Interactive chatbot interface built with Streamlit.

## Tech Stack

- **Language:** Python
- **RAG Framework:** Custom retrieval and generation pipeline
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers
- **LLM:** Llama 3.2 (3B)
- **Local LLM Runtime:** Ollama
- **Frontend:** Streamlit

## Run Locally

1. Clone the repository:
   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd RAG_Chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure [Ollama](https://ollama.com/) is installed and download the model:
   ```bash
   ollama pull llama3.2:3b
   ```

4. Index the BNSS documents:
   ```bash
   python ingest.py
   ```

5. Start the chatbot:
   ```bash
   python -m streamlit run app.py
   ```

6. Open `http://localhost:8501` in your browser.

## Disclaimer

This project is intended for educational and legal information purposes only. It does not replace advice from a qualified legal professional.

## Future Improvements

Planned improvements include better retrieval accuracy, more reliable legal citations, systematic evaluation, and enhanced answer quality.
