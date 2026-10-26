
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"
MODEL_NAME = "llama3.2:3b"

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection("bnss_legal_docs")


def ask(question):
    embedding = model.encode(
        question, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    context = ""
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        section = meta.get("section_number", "Unknown")
        title = meta.get("section_title", "")
        context += f"\n[BNSS s.{section} - {title}]\n{doc}\n"

    prompt = f"""You are a BNSS legal assistant.
Answer using only the provided context.
Cite relevant sections as [BNSS s.SECTION_NUMBER].
If the context does not contain enough information, say so.
Do not invent laws, sections, or citations.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    print("BNSS Legal RAG Chatbot (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            print("\nAssistant:", ask(question), "\n")
        except requests.RequestException as error:
            print(f" Ollama error: {error}\n")
