import fitz
from sentence_transformers import SentenceTransformers
pdf_path = 'data/pdfs/sample.pdf'
doc=ftiz.open(pdf_path)
text =""
for page in doc:
    text+=page.get_text()