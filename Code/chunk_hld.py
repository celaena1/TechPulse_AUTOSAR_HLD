import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Input_Data" / "extracted_hld_text.json"
OUTPUT_PATH = PROJECT_ROOT / "Input_Data" / "hld_chunks.json"
DOCUMENT_ID = "VH-GW-HLD-002"
DOCUMENT_VERSION = "2.0"
MIN_SECTION_WORDS = 35

FOOTER_PATTERN = re.compile(
    r"^VH-GW-HLD-002 Vehicle Gateway Control System HLD v2\.0$"
)
PAGE_FOOTER_PATTERN = re.compile(r"^Page \d+ of \d+$")


def normalized_line(text):
    return " ".join(text.split())


def section_root(section_number):
    if not section_number:
        return None
    return section_number.split(".", maxsplit=1)[0]


def section_label(section_number, section_title):
    if not section_number:
        return section_title
    return f"{section_number} {section_title}".strip()


def merge_small_adjacent_sections(sections):
    merged = []
    index = 0

    while index < len(sections):
        current = sections[index].copy()
        current["lines"] = list(current["lines"])
        current["pages"] = set(current["pages"])
        labels = [(current["section_number"], current["section_title"])]
        current_root = section_root(current["section_number"])

        while (
            current_root is not None
            and len(" ".join(current["lines"]).split()) < MIN_SECTION_WORDS
            and index + 1 < len(sections)
            and section_root(sections[index + 1]["section_number"]) == current_root
        ):
            index += 1
            following = sections[index]
            current["lines"].extend(following["lines"])
            current["pages"].update(following["pages"])
            labels.append(
                (following["section_number"], following["section_title"])
            )

        current["section_number"] = " + ".join(
            number for number, _ in labels if number
        )
        current["section_title"] = "; ".join(
            title for _, title in labels if title
        )
        merged.append(current)
        index += 1

    return merged


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Extracted HLD JSON was not found. Run extract_hld.py first: {INPUT_PATH}"
        )

    pages = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not pages:
        raise ValueError(f"No extracted pages found in {INPUT_PATH}")

    sections = []
    current = None

    def flush_current():
        if current is None:
            return
        text = " ".join(" ".join(current["lines"]).split())
        if text:
            sections.append(
                {
                    "section_number": current["section_number"],
                    "section_title": current["section_title"],
                    "lines": [text],
                    "pages": set(current["pages"]),
                }
            )

    for page_data in pages:
        page_number = page_data["page"]
        heading_map = {
            normalized_line(heading["text"]): heading
            for heading in page_data.get("headings", [])
        }

        for raw_line in page_data["text"].splitlines():
            line = normalized_line(raw_line)
            if not line or FOOTER_PATTERN.match(line) or PAGE_FOOTER_PATTERN.match(line):
                continue

            heading = heading_map.get(line)
            if heading:
                flush_current()
                current = {
                    "section_number": heading["number"],
                    "section_title": heading["title"],
                    "lines": [],
                    "pages": {page_number},
                }
                continue

            if current is None:
                current = {
                    "section_number": "",
                    "section_title": "Document opening",
                    "lines": [],
                    "pages": {page_number},
                }

            current["lines"].append(line)
            current["pages"].add(page_number)

    flush_current()
    sections = merge_small_adjacent_sections(sections)

    chunks = []
    section_chunk_counts = {}

    for section in sections:
        section_number = section["section_number"]
        section_chunk_counts[section_number] = (
            section_chunk_counts.get(section_number, 0) + 1
        )
        chunk_index = section_chunk_counts[section_number]
        safe_section_number = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            section_number or "opening",
        ).strip("_")
        chunk_id = (
            f"{DOCUMENT_ID}_{safe_section_number}_{chunk_index:02d}"
        )
        page_numbers = sorted(section["pages"])
        page_value = (
            page_numbers[0]
            if len(page_numbers) == 1
            else f"{page_numbers[0]}-{page_numbers[-1]}"
        )
        section_text = section_label(
            section_number,
            section["section_title"],
        )
        body = " ".join(section["lines"])
        chunk_text = (
            f"Section {section_text}. {body}"
            if section_number
            else body
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "page": page_value,
                "page_start": page_numbers[0],
                "page_end": page_numbers[-1],
                "section_number": section_number,
                "section_title": section["section_title"],
                "section": section_text,
                "document_id": DOCUMENT_ID,
                "document_version": DOCUMENT_VERSION,
                "text": chunk_text,
            }
        )

    if not chunks:
        raise ValueError("Chunking produced no chunks from the extracted HLD.")

    OUTPUT_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Pages processed: {len(pages)}")
    print(f"Logical section-aware chunks created: {len(chunks)}")
    print(f"Saved chunks to: {OUTPUT_PATH}")
    print("\n--- CHUNK INDEX ---")
    for chunk in chunks:
        print(
            f"{chunk['chunk_id']} | Page {chunk['page']} | "
            f"Section {chunk['section']} | "
            f"{len(chunk['text'].split())} words"
        )


if __name__ == "__main__":
    main()
