import nltk
import PyPDF2
from nltk.tokenize import sent_tokenize
from sklearn.mixture import GaussianMixture
from transformers import GPTSw3Tokenizer,GPT3Model


def extract_text(pdfs):
    all_text = ""
    for pdf_path in pdfs:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                all_text += page.extract_text()
    return all_text

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

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings



def raptor_indexing(embeddings,chunks):
    gmm = GaussianMixture(n_components=10,covariance_type='full',random_state=56)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    
    summaries = []

    for cluster in range(10):
        cluster_texts = [chunks[i] for i in range(len(chunks)) if labels[i] ==cluster]
        summary = 
        