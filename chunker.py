def chunk_pages(pages: list, chunk_size: int = 1000, overlap: int = 200) -> list:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk_text
                })
                chunk_id += 1

            start = end - overlap

    print(f"Total chunks created: {len(chunks)}")
    return chunks
