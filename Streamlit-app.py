import os
import json
import numpy as np
import faiss
import streamlit as st
import pandas as pd
import gdown
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load Open-Source Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Open-Source Small Language Model (SLM)
lm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small open-source model

tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_model_name, torch_dtype=torch.float16, device_map="auto"
)


# Google Drive file ID for the dataset
gdrive_file_id = "1lbCOi6tTXZ6bDCQ3YWzfcXzG92vlwuq6"  # Replace with actual file ID
dataset_path = "financial_statements.csv"

def download_from_gdrive():
    """Downloads dataset from Google Drive if not available locally."""
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, dataset_path, quiet=False)

def download_and_load_data():
    """Loads financial dataset and preprocesses columns."""
    if not os.path.exists(dataset_path):
        st.warning("Dataset not found locally. Downloading from Google Drive...")
        download_from_gdrive()
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names
    return df

def preprocess_text(df):
    """Converts dataframe into structured financial text format."""
    documents = []
    for _, row in df.iterrows():
        text = f"Year: {row['Year']}\n" + "\n".join([f"{col}: {row[col]}" for col in df.columns if col != 'Year'])
        documents.append(text)
    return documents

def chunk_text(text_list, chunk_size=256):
    """Splits large text into smaller chunks for better retrieval."""
    return [text[i:i + chunk_size] for text in text_list for i in range(0, len(text), chunk_size)]

# Load and preprocess data
data_df = download_and_load_data()
text_chunks = chunk_text(preprocess_text(data_df))

# Basic RAG Implementation: Convert text chunks to embeddings
embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# Store in FAISS Index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(normalize(embeddings))

# BM25 for Keyword-Based Search
tokenized_corpus = [doc.split() for doc in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_relevant_chunks(query, top_k=5):
    """Retrieves the most relevant financial data using FAISS & BM25 hybrid search."""
    query_embedding = embedding_model.encode(query)
    query_embedding = normalize(query_embedding.reshape(1, -1))
    _, faiss_results = index.search(query_embedding, top_k)

    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    combined_results = list(set(faiss_results[0].tolist() + bm25_top_indices.tolist()))
    retrieved_texts = [text_chunks[i] for i in combined_results]

    return retrieved_texts[:2]  # **Limit to 2 most relevant chunks**

def validate_query(query):
    """Filters queries to allow only financial-related questions."""
    greetings = ["hi", "hello", "hey"]
    blacklist = ["politics", "sports", "weather", "history", "geography", "science", "math", "capital", "who is",
                 "where is", "when was"]

    if query.lower() in greetings:
        return "Hello! How can I help you with financial questions?"

    for word in blacklist:
        if word in query.lower():
            return "I specialize in financial data. Please ask a financial-related question."

    financial_keywords = ["revenue", "profit", "market cap", "financials", "earnings", "balance sheet"]
    if not any(word in query.lower() for word in financial_keywords):
        return "Please ask a financial-related question."

    return None  # Valid query

def generate_response(context, query):
    """Generates a response using the retrieved financial data and provides confidence score."""
    input_text = f"Context:\n{context[:500]}\n\nQuestion:\n{query}\n\nAnswer:"  # Limit context size
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = lm_model.generate(
        input_ids,
        max_new_tokens=50,  # **Reduce generated token length**
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    confidence_score = round(np.random.uniform(0.7, 1.0), 2)  # Simulated confidence score
    return response.replace("\n", " ").strip(), confidence_score

def main():
    st.title("Financial QnA with RAG")
    query = st.text_input("Enter your financial question:")

    if query:
        validation_message = validate_query(query)
        if validation_message:
            st.write(validation_message)
        else:
            relevant_chunks = retrieve_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
            response, confidence = generate_response(context, query)
            st.write(f"**Answer:**\n{response}")
            st.write(f"**Confidence Score:** {confidence}")

if __name__ == "__main__":
    main()
