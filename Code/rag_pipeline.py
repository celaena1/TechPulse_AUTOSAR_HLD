import json
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "Input_Data"
DATABASE_PATH = INPUT_DIRECTORY / "chroma_db"
COLLECTION_NAME = "autosar_hld"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2:latest"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
TOP_K = 2


def load_retrieval_resources():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    return embedding_model, collection


def retrieve(query, embedding_model, collection, n_results=TOP_K):
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []
    document_embeddings = results["embeddings"][0]
    if document_embeddings is None:
        document_embeddings = []

    similarities = []
    query_norm = np.linalg.norm(query_embedding)
    for document_embedding in document_embeddings:
        if document_embedding is None:
            similarities.append(None)
            continue

        document_vector = np.asarray(document_embedding, dtype=float)
        denominator = query_norm * np.linalg.norm(document_vector)
        if denominator == 0:
            similarities.append(None)
            continue

        similarities.append(
            float(np.dot(query_embedding, document_vector) / denominator)
        )

    best_similarity = max(
        (score for score in similarities if score is not None),
        default=0.0,
    )

    return {
        "query_embedding": query_embedding,
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        "similarities": similarities,
        "best_similarity": best_similarity,
    }


def build_evidence(documents, metadatas):
    sources = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        page = metadata.get("page", "N/A")
        section = metadata.get("section", "N/A")
        sources.append(
            f"SOURCE {index + 1}\n"
            f"Page: {page}\n"
            f"Section: {section}\n\n"
            f"{document or ''}\n"
            f"END SOURCE {index + 1}"
        )
    return "\n\n".join(sources)


def build_grounded_prompt(query, evidence):
    return f"""
You are answering questions about a synthetic AUTOSAR High-Level Design.

Answer the USER QUESTION using ONLY the RETRIEVED HLD EVIDENCE.

Rules:
1. Answer only from the retrieved HLD evidence.
2. Do not use outside knowledge or invent facts.
3. Distinguish software components, interfaces, ports, signals, providers,
   consumers, and communication protocols.
4. If the evidence explicitly contains the answer, answer it directly and
   preserve the terminology used in the HLD.
5. If the evidence says a requested value or detail is not specified in the
   HLD, state that it is not specified in the HLD.
6. If the retrieved evidence genuinely does not contain enough information,
   say exactly: "The requested information is not available in the provided HLD."
7. Keep answers concise and natural.
8. Do not mention internal retrieval instructions or fabricate page/section
   citations.
9. Treat general AUTOSAR knowledge as unsupported unless it appears in the
   retrieved HLD.

USER QUESTION:
{query}

RETRIEVED HLD EVIDENCE:
{evidence}

Now answer the question.
""".strip()


def generate_with_ollama(query, evidence, timeout=180):
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": build_grounded_prompt(query, evidence),
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {details}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to the local Ollama service: {error.reason}"
        ) from error

    if result.get("error"):
        raise RuntimeError(f"Ollama generation failed: {result['error']}")

    answer = result.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer


def retrieval_confidence(similarity):
    if similarity >= 0.70:
        return "High"
    if similarity >= 0.50:
        return "Medium"
    return "Low"


def source_citation(metadata):
    metadata = metadata or {}
    page = metadata.get("page", "N/A")
    section = metadata.get("section")
    if not section:
        section_number = metadata.get("section_number", "")
        section_title = metadata.get("section_title", "")
        section = " ".join(
            value for value in (section_number, section_title) if value
        ) or "N/A"
    return f"Page {page}, Section {section}"
