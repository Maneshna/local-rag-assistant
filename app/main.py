import fitz
from sentence_transformers import SentenceTransformer



pdf_path = "data/pdfs/sample.pdf"

doc = fitz.open(pdf_path)

chunk_size = 500
overlap = 100



chunks = []

for page_num, page in enumerate(doc):

    page_text = page.get_text()

    start = 0

    while start < len(page_text):

        end = start + chunk_size

        chunk = page_text[start:end]

        # Store chunk + metadata
        chunks.append({
            "text": chunk,
            "source": "sample.pdf",
            "page": page_num + 1
        })

        # Move window with overlap
        start += chunk_size - overlap


chunk_texts = [chunk["text"] for chunk in chunks]


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



embeddings = model.encode(chunk_texts)



print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst Chunk:\n")
print(chunks[0])

print("\nFirst Embedding Vector Length:")
print(len(embeddings[0]))