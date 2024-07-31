import PyPDF2
from Functions import chunk_text, extract_text, embed_chunks,raptor_indexing


pdf_files = [
    r"F:\\machine learning\\Model_building\\Textbook_Mining\Atomic Habits James Clear.pdf",
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\Cant Hurt Me_ Master Your Mind and Defy the Odds.pdf',
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\maths_text1.pdf'
]

extracted_text = extract_text(pdf_files)

chunks = chunk_text(extract_text)
embeddings = embed_chunks(chunks)
summaries,summary_embeddings = raptor_indexing(embeddings,chunks)
# print(summaries)
# print(summary_embeddings)




