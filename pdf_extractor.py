import fitz  # this is pymupdf

def extract_pdf_pages(pdf_path: str) -> list:
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():  # only add if page has text
                pages.append({
                    "source": pdf_path,
                    "page": page_num + 1,
                    "text": text
                })
    
    print(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages
