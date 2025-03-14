Financial QnA with RAG

Overview

This project implements a Retrieval-Augmented Generation (RAG) model to answer financial questions based on company financial statements from the last two years. The system is built using FAISS, BM25, Sentence Transformers, and a small open-source LLM (OPT-1.3B).

Features

Hybrid Search: Combines FAISS (vector search) and BM25 (keyword search) for enhanced retrieval.

Memory-Augmented Retrieval: Uses prior queries to improve results.

Guardrails: Ensures only financial-related queries are processed.

Streamlit UI: Provides an interactive interface for user queries and chat history.

Confidence Score: Each response includes a reliability score.

Technologies Used

Vector Search: FAISS

Keyword Search: BM25

Embeddings Model: all-MiniLM-L6-v2 (Sentence Transformers)

LLM: Facebook OPT-1.3B

Frameworks: PyTorch, Hugging Face Transformers, Streamlit

Storage: Google Drive for financial dataset

Setup Instructions

1. Clone Repository & Install Dependencies

git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt

2. Run the Streamlit App

streamlit run Streamlit-app.py

Dataset

The system retrieves financial data from a CSV file stored in Google Drive.

The dataset contains financial statements of multiple companies over the past two years.

How It Works

1. Data Collection & Preprocessing

Loads financial data from Google Drive.

Structures it into retrievable text chunks.

2. Retrieval-Augmented Generation (RAG)

Step 1: Retrieve relevant financial data using FAISS (vector search) & BM25 (keyword search).

Step 2: Pass retrieved content & user query to the LLM to generate an answer.

3. Guardrail Implementation

Input-Side: Blocks non-financial queries (e.g., sports, politics).

4. Chat History

Stores past interactions in session state to enable conversational memory.

Example Queries

High-Confidence Query:

Q: "What was Apple's net profit in 2022?"
A: "Apple's net profit in 2022 was $99,803 million."
Confidence Score: 0.95

Low-Confidence Query:

Q: "What was Tesla's EBITDA in 2023?"
A: "Data not available."
Confidence Score: 0.5

Irrelevant Query:

Q: "Who is Virat Kohli?"
A: "I specialize in financial data. Please ask a financial-related question."

Deployment

This app can be deployed on Streamlit Cloud using the following steps:

Push the repository to GitHub.

Go to Streamlit Cloud and connect the GitHub repository.

Set up environment variables and dependencies.

Deploy and get a shareable app link!

Future Improvements

Output-Side Guardrails to filter LLM hallucinations.

Fine-tuning retrieval methods for better accuracy.

Adding more financial metrics to enhance responses.

Contributors

Amandeep Singh - Developer