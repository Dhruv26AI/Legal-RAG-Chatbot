
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bnss_chunks.jsonl"
DB_PATH = BASE_DIR / "chroma_db"

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(
    name="bnss_legal_docs",
    metadata={"hnsw:space": "cosine"}
)

with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

for start in range(0, len(chunks), 64):
    batch = chunks[start:start + 64]
    documents = [item["text"] for item in batch]
    ids = [item["chunk_id"] for item in batch]
    metadatas = [
        {
            "section_number": str(item.get("section_number") or ""),
            "section_title": str(item.get("section_title") or ""),
            "subsection": str(item.get("subsection") or ""),
            "page_start": int(item.get("page_start") or 0)
        }
        for item in batch
    ]
    embeddings = model.encode(
        documents, normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

print(f"Indexed {collection.count()} BNSS chunks in ChromaDB.")
