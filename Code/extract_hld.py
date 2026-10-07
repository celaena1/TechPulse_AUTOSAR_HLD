import json
import re
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "Input_Data"
PDF_PATH = INPUT_DIRECTORY / (
    "VH-GW-HLD-002 Vehicle Gateway Control System HLD v2.0.pdf"
)
OUTPUT_PATH = INPUT_DIRECTORY / "extracted_hld_text.json"

HEADING_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$"
)


def detect_heading(text, font_size, is_bold):
    match = HEADING_PATTERN.match(text.strip())
    if not match:
        return None

    number = match.group("number")
    title = match.group("title").strip()
    is_main_heading = "." not in number and font_size >= 15
    is_subheading = "." in number and (font_size >= 13 or is_bold)

    if not (is_main_heading or is_subheading):
        return None

    return {
        "text": text.strip(),
        "number": number,
        "title": title,
    }


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Source HLD PDF was not found: {PDF_PATH}"
        )

    pages = []
    document = pymupdf.open(PDF_PATH)

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            headings = []

            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(span["text"] for span in spans).strip()
                    if not line_text:
                        continue

                    font_size = max(
                        (span.get("size", 0) for span in spans),
                        default=0,
                    )
                    is_bold = any(
                        span.get("flags", 0) & 16
                        for span in spans
                    )
                    heading = detect_heading(line_text, font_size, is_bold)
                    if heading:
                        headings.append(heading)

            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                    "headings": headings,
                }
            )
    finally:
        document.close()

    OUTPUT_PATH.write_text(
        json.dumps(pages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Source PDF: {PDF_PATH}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Output: {OUTPUT_PATH}")
    print("Detected section headings:")
    for page in pages:
        for heading in page["headings"]:
            print(
                f"  Page {page['page']}: "
                f"{heading['number']} {heading['title']}"
            )


if __name__ == "__main__":
    main()
