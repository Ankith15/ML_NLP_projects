import PyPDF2

def extract_text(pdfs):
    all_text = ""
    for pdf_path in pdfs:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                all_text += page.extract_text()
    return all_text

pdf_files = [
    r"F:\\machine learning\\Model_building\\Textbook_Mining\Atomic Habits James Clear.pdf",
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\Cant Hurt Me_ Master Your Mind and Defy the Odds.pdf',
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\maths_text1.pdf'
]

extracted_text = extract_text(pdf_files)

print(extracted_text)


import nltk
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
