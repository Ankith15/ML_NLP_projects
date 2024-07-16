import PyPDF2

def extract_text(pdfs):
    all_text = ""
    for pdf_path in pdfs:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfFileReader(f)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                all_text += page.extract_text()
    return all_text

# List of PDF file paths
pdf_files = [r"F:\\machine learning\\Model_building\\Textbook_Mining\Atomic Habits James Clear.pdf", r'Textbook_Minig/Cant Hurt Me_ Master Your Mind and Defy the Odds.pdf', r'Textbook_Minig/maths_text1.pdf']

# Extract text from all PDFs
extracted_text = extract_text(pdf_files)

# Print or save the extracted text
print(extracted_text)
