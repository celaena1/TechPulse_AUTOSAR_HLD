import sys

from rag_pipeline import (
    TOP_K,
    build_evidence,
    generate_with_ollama,
    load_retrieval_resources,
    retrieve,
    retrieval_confidence,
    source_citation,
)


def main():
    print("Loading the HLD retrieval pipeline...")
    embedding_model, collection = load_retrieval_resources()

    query = input("\nEnter your question: ").strip()
    if not query:
        print("Please enter a question.")
        return

    retrieved = retrieve(
        query,
        embedding_model,
        collection,
        n_results=TOP_K,
    )
    documents = retrieved["documents"]
    metadatas = retrieved["metadatas"]
    evidence = build_evidence(documents, metadatas)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    try:
        answer = generate_with_ollama(query, evidence)
        response_mode = "LLM-generated"
        print(answer)
    except Exception as error:
        response_mode = "Evidence fallback"
        print(f"LLM unavailable: {error}", file=sys.stderr)
        print("LLM unavailable — showing retrieved HLD evidence.\n")
        print(evidence)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for metadata in metadatas:
        print(f"- {source_citation(metadata)}")

    similarity = retrieved["best_similarity"]
    print("\n" + "=" * 70)
    print("RETRIEVAL CONFIDENCE")
    print("=" * 70)
    print(f"Retrieval confidence: {retrieval_confidence(similarity)}")
    print(f"Similarity score: {similarity:.2f}")
    print("This is a retrieval similarity indicator, not a probability of answer correctness.")
    print(f"\nResponse mode: {response_mode}")


if __name__ == "__main__":
    main()
