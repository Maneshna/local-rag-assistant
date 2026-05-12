import fitz
from sentence_transformers import SentenceTransformer

pdf_path = "data/pdfs/sample.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]

    chunks.append(chunk)

    start += chunk_size - overlap

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

print(embeddings.shape)