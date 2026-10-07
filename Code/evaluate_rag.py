import re
import sys

from rag_pipeline import (
    TOP_K,
    build_evidence,
    generate_with_ollama,
    load_retrieval_resources,
    retrieve,
    source_citation,
)


sys.stdout.reconfigure(errors="backslashreplace")

EVALUATION_QUESTIONS = [
    {
        "question": "Which component provides VehicleSpeed?",
        "required_terms": ["SensorInterface"],
    },
    {
        "question": "Which component consumes VehicleSpeed?",
        "required_terms": ["VehicleControlComponent"],
    },
    {
        "question": "What interface contains VehicleSpeed?",
        "required_terms": ["VehicleDataInterface"],
    },
    {
        "question": "Which component provides BrakeStatus?",
        "required_terms": ["VehicleControlComponent"],
    },
    {
        "question": "Which component consumes BrakeStatus?",
        "required_terms": ["BrakeControlComponent"],
    },
    {
        "question": (
            "Which communication protocol is primary for the explicit HLD signals?"
        ),
        "required_terms": ["CAN"],
    },
    {
        "question": (
            "What is the relationship between DiagnosticManager and "
            "CommunicationManager?"
        ),
        "required_terms": [
            "DiagnosticInterface",
            "DiagnosticRequest",
            "DiagnosticResponse",
        ],
        "required_phrases": [
            "CommunicationManager provides DiagnosticRequest to DiagnosticManager",
            "DiagnosticManager provides DiagnosticResponse to CommunicationManager",
        ],
    },
    {
        "question": "Which component provides DiagnosticResponse?",
        "required_terms": ["DiagnosticManager"],
    },
    {
        "question": "What is the UDS service ID for reading a DTC?",
        "not_specified": True,
    },
    {
        "question": "What is the ECU address used for DiagnosticManager?",
        "not_specified": True,
    },
    {
        "question": "What NRC is returned for an unsupported diagnostic request?",
        "not_specified": True,
    },
    {
        "question": "What is the exact SOME/IP service ID?",
        "not_specified": True,
    },
]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def grade_answer(answer, evaluation_case):
    normalized_answer = normalize(answer)

    if evaluation_case.get("not_specified"):
        refusal_phrases = (
            "not specified",
            "not available in the provided hld",
            "not available in provided hld",
        )
        if any(phrase in normalized_answer for phrase in refusal_phrases):
            return "UNSUPPORTED/REFUSAL ANSWER (correct)", True
        return "INCORRECT", False

    required_terms = evaluation_case.get("required_terms", [])
    required_phrases = evaluation_case.get("required_phrases", [])
    terms_present = all(
        normalize(term) in normalized_answer for term in required_terms
    )
    phrases_present = all(
        normalize(phrase) in normalized_answer for phrase in required_phrases
    )

    if terms_present and phrases_present:
        return "CORRECT", True
    return "INCORRECT", False


def main():
    embedding_model, collection = load_retrieval_resources()
    print("=" * 78)
    print("AUTOSAR HLD v2.0 — END-TO-END RAG EVALUATION")
    print("=" * 78)
    print(f"Collection: {collection.name} ({collection.count()} chunks)")
    print(f"Retrieval: dense Top-{TOP_K}")
    print("LLM: local Ollama llama3.2:latest, temperature=0\n")

    total = 0
    correct = 0
    incorrect = 0
    unsupported_refusal = 0

    for question_number, case in enumerate(EVALUATION_QUESTIONS, start=1):
        query = case["question"]
        retrieved = retrieve(
            query,
            embedding_model,
            collection,
            n_results=TOP_K,
        )
        documents = retrieved["documents"]
        metadatas = retrieved["metadatas"]

        print("-" * 78)
        print(f"Question {question_number}: {query}")
        print("Retrieved evidence/source:")
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            print(f"\n  Source {index + 1}: {source_citation(metadata)}")
            print(f"  {document}")

        if not documents:
            answer = "No evidence was returned by ChromaDB."
            result_label = "INCORRECT"
            passed = False
        else:
            evidence = build_evidence(documents, metadatas)
            try:
                answer = generate_with_ollama(query, evidence)
            except Exception as error:
                print(f"\nLLM evaluation error: {error}", file=sys.stderr)
                answer = f"GENERATION ERROR: {error}"
                result_label = "INCORRECT"
                passed = False
            else:
                result_label, passed = grade_answer(answer, case)

        print(f"\nLLM answer: {answer}")
        print(
            "Expected answer: "
            + (
                "Not specified in the HLD"
                if case.get("not_specified")
                else "; ".join(
                    case.get("required_phrases", [])
                    + case.get("required_terms", [])
                )
            )
        )
        print(f"Result: {result_label} — {'PASS' if passed else 'FAIL'}")

        total += 1
        if passed:
            correct += 1
        else:
            incorrect += 1
        if result_label.startswith("UNSUPPORTED/REFUSAL"):
            unsupported_refusal += 1

    accuracy = (correct / total * 100) if total else 0.0
    print("\n" + "=" * 78)
    print("EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(
        "Unsupported/refusal answers (correct for specified out-of-scope "
        f"questions): {unsupported_refusal}"
    )
    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
