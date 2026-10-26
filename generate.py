
import requests

MODEL_NAME = "llama3.2:3b"


def generate_answer(question, chunks):
    if not chunks:
        return "I could not find relevant information in the BNSS database."

    context = "\n\n".join(
        f"[BNSS s.{item['metadata'].get('section_number', 'Unknown')}]\n"
        f"{item['text']}"
        for item in chunks
    )

    prompt = f"""You are a BNSS legal information assistant.
Answer using only the supplied context.
Cite sections using [BNSS s.SECTION_NUMBER].
Never invent legal provisions or section numbers.
If the context is insufficient, say so.
Explain the law in simple language.

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
            "stream": False,
            "options": {"num_predict": 200}
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]
