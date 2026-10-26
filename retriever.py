
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)
collection = client.get_collection("bnss_legal_docs")


def retrieve(question, top_k=5):
    embedding = model.encode(
        question, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    ]
