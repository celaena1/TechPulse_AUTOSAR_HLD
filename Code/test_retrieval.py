import sys

from rag_pipeline import (
    TOP_K,
    load_retrieval_resources,
    retrieve,
    source_citation,
)


sys.stdout.reconfigure(errors="backslashreplace")

QUESTIONS = [
    "Which component provides VehicleSpeed?",
    "Which component consumes VehicleSpeed?",
    "What interface contains VehicleSpeed?",
    "Which component provides BrakeStatus?",
    "Which component consumes BrakeStatus?",
    "What is the relationship between DiagnosticManager and CommunicationManager?",
    "Which component provides DiagnosticResponse?",
    "Which communication protocol is primary for all five explicit HLD signals?",
    "What is the UDS service ID for reading a DTC?",
    "What ECU address is specified for DiagnosticManager?",
    "What NRC is specified for an unsupported diagnostic request?",
    "What is the exact SOME/IP service ID?",
]


def main():
    embedding_model, collection = load_retrieval_resources()
    print(f"Collection: {collection.name} ({collection.count()} chunks)")
    print(f"Retrieval depth: Top-{TOP_K}\n")

    for question_number, query in enumerate(QUESTIONS, start=1):
        retrieved = retrieve(
            query,
            embedding_model,
            collection,
            n_results=TOP_K,
        )

        print("=" * 100)
        print(f"QUESTION {question_number}: {query}")
        print("=" * 100)

        if not retrieved["documents"]:
            print("No results returned by ChromaDB.\n")
            continue

        for index, document in enumerate(retrieved["documents"]):
            metadata = retrieved["metadatas"][index]
            distance = (
                retrieved["distances"][index]
                if index < len(retrieved["distances"])
                else None
            )
            similarity = (
                retrieved["similarities"][index]
                if index < len(retrieved["similarities"])
                else None
            )

            print(f"\nRank {index + 1}")
            print(f"Source: {source_citation(metadata)}")
            print(f"Chroma distance: {distance if distance is not None else 'N/A'}")
            print(
                "Cosine similarity: "
                f"{similarity:.4f}" if similarity is not None
                else "Cosine similarity: N/A"
            )
            print("Retrieved chunk:")
            print("-" * 80)
            print(document)
            print("-" * 80)

        print()


if __name__ == "__main__":
    main()
