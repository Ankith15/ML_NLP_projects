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
