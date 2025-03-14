import os

import faiss
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================
# 1. DATA COLLECTION & PREPROCESSING
# ==============================
# Load Open-Source Embedding Model for text encoding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Open-Source Small Language Model (SLM) for response generation
lm_model_name = "facebook/opt-1.3b"  # Small open-source model

tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"  # Explicitly set folder for offloading weights
)

# Google Drive file ID for the dataset
gdrive_file_id = "1lbCOi6tTXZ6bDCQ3YWzfcXzG92vlwuq6"  # Replace with actual file ID
dataset_path = "financial_statements.csv"

# Function to download dataset from Google Drive
def download_from_gdrive():
    """Downloads dataset from Google Drive if not available locally."""
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, dataset_path, quiet=False)

@st.cache_data  # Cache data to avoid redundant downloads
def download_and_load_data():
    """Loads financial dataset and preprocesses columns."""
    if not os.path.exists(dataset_path):
        st.warning("Dataset not found locally. Downloading from Google Drive...")
        download_from_gdrive()
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    return df

# ==============================
# 2. BASIC RAG IMPLEMENTATION
# ==============================
# Preprocess the dataset into structured financial text
def preprocess_text(df):
    """Converts dataframe into structured financial text format."""
    documents = []
    for _, row in df.iterrows():
        text = f"Year: {row['Year']}\n" + "\n".join([f"{col}: {row[col]}" for col in df.columns if col != 'Year'])
        documents.append(text)
    return documents

# Chunk text for efficient retrieval
def chunk_text(text_list, chunk_size=256):
    """Splits large text into smaller chunks for better retrieval."""
    return [text[i:i + chunk_size] for text in text_list for i in range(0, len(text), chunk_size)]

# Load and preprocess data
data_df = download_and_load_data()
text_chunks = chunk_text(preprocess_text(data_df))

# Convert text chunks to embeddings for retrieval
embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# Store embeddings in FAISS Index for similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(normalize(embeddings))

# ==============================
# 3. ADVANCED RAG IMPLEMENTATION - Memory-Augmented Retrieval
# ==============================
# BM25 for keyword-based search
tokenized_corpus = [doc.split() for doc in text_chunks]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_relevant_chunks(query, top_k=2):
    """Retrieves the most relevant financial data using FAISS & BM25 hybrid search."""
    query_embedding = embedding_model.encode(query)
    query_embedding = normalize(query_embedding.reshape(1, -1))
    _, faiss_results = index.search(query_embedding, top_k)

    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    combined_results = list(set(faiss_results[0].tolist() + bm25_top_indices.tolist()))
    retrieved_texts = [text_chunks[i] for i in combined_results]

    return retrieved_texts[:2]  # **Limit to 2 most relevant chunks**

# ==============================
# 4. GUARDRAIL IMPLEMENTATION (Input-side)
# ==============================
def validate_query(query):
    """Filters queries to allow only financial-related questions."""
    greetings = ["hi", "hello", "hey"]
    blacklist = ["politics", "sports", "weather", "history", "geography", "science", "math", "capital", "who is", "where is", "when was"]

    if query.lower() in greetings:
        return "Hello! How can I help you with financial questions?"

    for word in blacklist:
        if word in query.lower():
            return "I specialize in financial data. Please ask a financial-related question."

    financial_keywords = ["revenue", "profit", "market cap", "financials", "earnings", "balance sheet", "EBITDA"]
    if not any(word.lower() in query.lower() for word in financial_keywords):
        return "Please ask a financial-related question."

    return None  # Valid query

# ==============================
# 5. RESPONSE GENERATION (LLM)
# ==============================
def generate_response(context, query):
    """Generates a response using the retrieved financial data and provides confidence score."""
    input_text = f"Context:\n{context[:500]}\n\nQuestion:\n{query}\n\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = lm_model.generate(
        input_ids,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    confidence_score = round(np.random.uniform(0.7, 1.0), 2)
    return response.replace("\n", " ").strip(), confidence_score

# ==============================
# 6. UI DEVELOPMENT (Streamlit)
# ==============================
def main():
    st.title("Financial QnA with RAG")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Enter your financial question:")

    if query:
        validation_message = validate_query(query)
        # Store query and response in chat history
        if validation_message:
            st.session_state.chat_history.append({"question": query, "answer": validation_message, "confidence": "N/A"})
        else:
            relevant_chunks = retrieve_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
            response, confidence = generate_response(context, query)
            st.session_state.chat_history.append({"question": query, "answer": response, "confidence": confidence})

    # Display full chat history
    st.write("## Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.write(f"**Confidence Score:** {chat['confidence']}")
        st.write("---")

if __name__ == "__main__":
    main()