import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import numpy as np
import mysql.connector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",   
        password="Festus3004.",
        database="vector_db"
    )

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings and store them in the database
def store_embeddings(text_chunks):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    connection = get_db_connection()
    cursor = connection.cursor()
    for chunk in text_chunks:
        embedding_vector = embeddings_model.embed(chunk)
        embedding_blob = np.array(embedding_vector, dtype=np.float32).tobytes()
        cursor.execute(
            "INSERT INTO embeddings (document_text, embedding) VALUES (%s, %s)",
            (chunk, embedding_blob)
        )
    connection.commit()
    cursor.close()
    connection.close()

# Function to retrieve embeddings from the database
def fetch_embeddings():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT document_text, embedding FROM embeddings")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    documents = []
    embeddings = []
    for doc_text, embedding_blob in results:
        embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
        documents.append(doc_text)
        embeddings.append(embedding_vector)
    return documents, embeddings

# Function to perform similarity search
def similarity_search(query, documents, embeddings):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_embedding = embeddings_model.embed(query)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    most_similar_idx = np.argmax(similarities)
    return documents[most_similar_idx]

# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "Answer is not available in the context." Do not provide incorrect information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    documents, embeddings = fetch_embeddings()
    relevant_doc = similarity_search(user_question, documents, embeddings)
    chain = get_conversational_chain()
    response = chain({"input_documents": [relevant_doc], "question": user_question}, return_only_outputs=True)
    st.write("Response:", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with PDF Using Gemini")

    user_question = st.text_input("Ask a question about your PDFs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                store_embeddings(text_chunks)
                st.success("Processing complete")

if __name__ == "__main__":
    main()
