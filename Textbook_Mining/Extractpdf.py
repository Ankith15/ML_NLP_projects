import PyPDF2
from Functions import chunk_text, extract_text


pdf_files = [
    r"F:\\machine learning\\Model_building\\Textbook_Mining\Atomic Habits James Clear.pdf",
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\Cant Hurt Me_ Master Your Mind and Defy the Odds.pdf',
    r'F:\\machine learning\\Model_building\\Textbook_Mining\\maths_text1.pdf'
]

extracted_text = extract_text(pdf_files)

print(extracted_text)






