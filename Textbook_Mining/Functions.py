import nltk
import numpy as np
import PyPDF2
from nltk.tokenize import sent_tokenize
from sklearn.mixture import GaussianMixture
from transformers import GPTSw3Tokenizer,GPT3Model
from transformers import BartTokenizer, BartForConditionalGeneration



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


bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def bart_summarize(texts):
    full_text = " ".join(texts)
    
    inputs = bart_tokenizer.encode("summarize: " + full_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = bart_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary




def raptor_indexing(embeddings,chunks):
    gmm = GaussianMixture(n_components=10,covariance_type='full',random_state=56)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    
    summaries = []

    for cluster in range(10):
        cluster_texts = [chunks[i] for i in range(len(chunks)) if labels[i] ==cluster]
        summary = bart_summarize(cluster_texts)
        summaries.append(summary)
    
    summary_embeddings = model.encode(summaries,convert_to_tensor=True)
    return summaries,summary_embeddings

def insert_embeddings(collection, embeddings):
    ids = [i for i in range(len(embeddings))]
    collection.insert([ids, embeddings])
def insert_embeddings(collection, embeddings):
    ids = [i for i in range(len(embeddings))]
    collection.insert([ids, embeddings])

# Example embeddings (Replace with actual embeddings)
example_embeddings = np.random.rand(10, 384).tolist()

# insert_embeddings(collection, example_embeddings)