import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "Input_Data" / "hld_chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "Input_Data" / "hld_embeddings.json"
MODEL_NAME = "all-MiniLM-L6-v2"
DOCUMENT_ID = "VH-GW-HLD-002"
DOCUMENT_VERSION = "2.0"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Chunk file was not found. Run chunk_hld.py first: {CHUNKS_PATH}"
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique before embedding.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match the current chunk count."
        )

    output = {
        "document_id": DOCUMENT_ID,
        "document_version": DOCUMENT_VERSION,
        "embedding_model": MODEL_NAME,
        "chunks_sha256": hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest(),
        "entries": [
            {
                "chunk_id": chunk["chunk_id"],
                "text_sha256": sha256_text(chunk["text"]),
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ],
    }

    EMBEDDINGS_PATH.write_text(
        json.dumps(output, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
