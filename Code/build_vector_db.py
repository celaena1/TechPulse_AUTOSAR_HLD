import hashlib
import json
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "Input_Data"
CHUNKS_PATH = INPUT_DIRECTORY / "hld_chunks.json"
EMBEDDINGS_PATH = INPUT_DIRECTORY / "hld_embeddings.json"
DATABASE_PATH = INPUT_DIRECTORY / "chroma_db"
COLLECTION_NAME = "autosar_hld"
DOCUMENT_ID = "VH-GW-HLD-002"
DOCUMENT_VERSION = "2.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Chunk file was not found. Run chunk_hld.py first: {CHUNKS_PATH}"
        )
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(
            "Embedding file was not found. Run create_embeddings.py first: "
            f"{EMBEDDINGS_PATH}"
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embedding_data = json.loads(
        EMBEDDINGS_PATH.read_text(encoding="utf-8")
    )

    if embedding_data.get("document_id") != DOCUMENT_ID:
        raise ValueError("Embeddings do not belong to VH-GW-HLD-002.")
    if embedding_data.get("document_version") != DOCUMENT_VERSION:
        raise ValueError("Embeddings are not from HLD version 2.0.")
    if embedding_data.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError(
            f"Expected embeddings from {EMBEDDING_MODEL}."
        )

    current_chunks_hash = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    if embedding_data.get("chunks_sha256") != current_chunks_hash:
        raise ValueError(
            "The chunks changed after embedding. Run create_embeddings.py again."
        )

    entries = embedding_data.get("entries", [])
    if len(chunks) != len(entries):
        raise ValueError(
            "Chunk count does not match the embedding count."
        )

    for chunk, entry in zip(chunks, entries):
        if chunk["chunk_id"] != entry.get("chunk_id"):
            raise ValueError(
                "Chunk order/IDs do not match the saved embeddings."
            )
        if sha256_text(chunk["text"]) != entry.get("text_sha256"):
            raise ValueError(
                f"Embedding text does not match chunk {chunk['chunk_id']}."
            )

    client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    existing_collections = {
        collection.name for collection in client.list_collections()
    }
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "document_id": DOCUMENT_ID,
            "document_version": DOCUMENT_VERSION,
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section": chunk["section"],
            "section_number": chunk["section_number"],
            "section_title": chunk["section_title"],
            "document_id": chunk["document_id"],
            "document_version": chunk["document_version"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[entry["embedding"] for entry in entries],
        metadatas=metadatas,
    )

    stored_count = collection.count()
    if stored_count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} stored chunks, found {stored_count}."
        )

    print("ChromaDB collection rebuilt from scratch.")
    print(f"Collection: {collection.name}")
    print(f"Document: {DOCUMENT_ID}, version {DOCUMENT_VERSION}")
    print(f"Documents stored: {stored_count}")
    print(f"Database location: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
