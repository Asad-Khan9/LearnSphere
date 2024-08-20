import streamlit as st
import PyPDF2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import io
from app import parse_mcq_questions
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def encode_texts(query, documents):
    # Combine the query and documents to build a consistent vocabulary
    corpus = [query] + documents
    
    # Initialize and fit the vectorizer on the entire corpus
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    
    # Transform the query and documents using the same vectorizer
    query_embedding = vectorizer.transform([query]).toarray()
    doc_embeddings = vectorizer.transform(documents).toarray()
    
    return query_embedding, doc_embeddings

def retrieve_documents(query, documents, top_k=3):
    # Encode the query and documents to ensure consistent dimensions
    query_embedding, doc_embeddings = encode_texts(query, documents)

    # Debugging: Print the shapes of the embeddings
    print(f"Query Embedding Shape: {query_embedding.shape}")
    print(f"Document Embedding Shape: {doc_embeddings.shape}")

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

    # Retrieve the top k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def stream_response(prompt, placeholder):
    response = ""
    for chunk in generate_llama_response(prompt,True):
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            placeholder.markdown(response + "▌")
    placeholder.markdown(response)
    return response

def summarize_pdf(documents, placeholder):
    context = " ".join(documents[:3])
    prompt = f"""Provide a concise summary of the main points in the following text:

Text to summarize: {context}

Summary:"""
    return stream_response(prompt, placeholder)

def generate_llama_response(prompt,stream):
    messages = [
        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
        {"role": "user", "content": prompt}
    ]
    
    completion = client.chat.completions.create(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        messages=messages,
        temperature=0.7,
        stream=False,
    )
    if stream:
        return completion
    else :
        return completion.choices[0].message.content

def create_mcq_questions(documents):
    context = " ".join(documents[:3])
    prompt = f"""Generate 5 multiple-choice questions based on the following text. Each question should have 4 options (A, B, C, D) with one correct answer. Format your response as follows for each question:

        Question: [Question text]
        options: [A,B,C,D]
        Correct Answer: [A]

        Text to create questions from: {context}

        Questions:"""
   
    response = generate_llama_response(prompt,False)
    parse_mcq_questions(response)



    


def create_notes(documents, placeholder):
    context = " ".join(documents[:3])
    prompt = f"""Create concise and informative notes based on the following text. Focus on key concepts, definitions, and important points:

Text to create notes from: {context}

Notes:"""
    return stream_response(prompt, placeholder)
# import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
