import streamlit as st
import PyPDF2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import io
st.set_page_config(page_title="PDF Analysis with Llama 2", page_icon="📚")
from helperFunctions import *

# Point to the local server


# ... (keep all the other functions unchanged)
def parse_mcq_questions(response):
    questions = []
    quest_answr = {}
    n = 1

    for line in response.split('\n'):
        if line.startswith(f'Question {n}:'):
            if quest_answr:
                questions.append(quest_answr)
            quest_answr = {"question": line.split(f'Question {n}: ')[1], "options": {}, "correct_answer": ""}
            n += 1 
        elif line.startswith('Options:'):
            options = line.split('Options: ')[1].split(' ')
            quest_answr['options'] = {
                "A": " ".join(options[1:options.index('B)')]),
                "B": " ".join(options[options.index('B)') + 1:options.index('C)')]),
                "C": " ".join(options[options.index('C)') + 1:options.index('D)')]),
                "D": " ".join(options[options.index('D)') + 1:])
            }
        elif line.startswith('Correct Answer:'):
            correct_ans = line.split('Correct Answer: ')[1]
            correct_ans_letter = correct_ans[0]
            quest_answr["correct_answer"] = quest_answr["options"][correct_ans_letter]

    if quest_answr:
        questions.append(quest_answr)
    for i, q in enumerate(questions):
        st.write(f"Question {i + 1}: {q['question']}")
        
        options = list(q['options'].values())
        options.append("Clear response")
        
        selected_option = st.radio('Choose an option:', options, key=f"question_{i}", index=len(options)-1)
        
        if selected_option == "Clear response":
            st.session_state[f"answer_{i}"] = None
        else:
            st.session_state[f"answer_{i}"] = selected_option

    st.write('---')

    if st.button('Submit'):
        for i, q in enumerate(questions):
            selected_answer = st.session_state.get(f"answer_{i}")
            if selected_answer:
                if selected_answer == q["correct_answer"]:
                    st.success(f"Question {i+1}: Correct!")
                else:
                    st.error(f"Question {i+1}: Wrong!")
                st.write(f"The correct answer is: {q['correct_answer']}")
            else:
                st.warning(f"Question {i+1}: No answer selected")
        
        st.write('---') 

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


def parse_mcq_questions(response):
    questions = []
    quest_answr = {}
    n = 1

    for line in response.split('\n'):
        if line.startswith(f'Question {n}:'):
            if quest_answr:
                questions.append(quest_answr)
            quest_answr = {"question": line.split(f'Question {n}: ')[1], "options": {}, "correct_answer": ""}
            n += 1
        elif line.startswith('Options:'):
            options = line.split('Options: ')[1].split(' ')
            quest_answr['options'] = {
                "A": " ".join(options[1:options.index('B)')]),
                "B": " ".join(options[options.index('B)') + 1:options.index('C)')]),
                "C": " ".join(options[options.index('C)') + 1:options.index('D)')]),
                "D": " ".join(options[options.index('D)') + 1:])
            }
        elif line.startswith('Correct Answer:'):
            correct_ans = line.split('Correct Answer: ')[1]
            correct_ans_letter = correct_ans[0]
            quest_answr["correct_answer"] = quest_answr["options"][correct_ans_letter]

    if quest_answr:
        questions.append(quest_answr)
    for i, q in enumerate(questions):
        st.write(f"Question {i + 1}: {q['question']}")
        
        options = list(q['options'].values())
        options.append("Clear response")
        
        selected_option = st.radio('Choose an option:', options, key=f"question_{i}", index=len(options)-1)
        
        if selected_option == "Clear response":
            st.session_state[f"answer_{i}"] = None
        else:
            st.session_state[f"answer_{i}"] = selected_option

    st.write('---')

    if st.button('Submit'):
        for i, q in enumerate(questions):
            selected_answer = st.session_state.get(f"answer_{i}")
            if selected_answer:
                if selected_answer == q["correct_answer"]:
                    st.success(f"Question {i+1}: Correct!")
                else:
                    st.error(f"Question {i+1}: Wrong!")
                st.write(f"The correct answer is: {q['correct_answer']}")
            else:
                st.warning(f"Question {i+1}: No answer selected")
        
        st.write('---') 
    


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

# Streamlit App
def main():
    st.title("PDF Analysis with Llama 2")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            chunk_size = 1000
            documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            st.success("PDF uploaded and processed successfully!")

            tabs = st.tabs(["Summarize PDF", "Generate MCQ Questions", "Create Notes"])

            with tabs[0]:
                if st.button("Summarize PDF"):
                    with st.spinner("Generating summary..."):
                        placeholder = st.empty()
                        summarize_pdf(documents, placeholder)

            with tabs[1]:  # Tab for Generating MCQ Questions
                if st.button("Generate MCQ Questions"):
                    with st.spinner("Generating MCQ questions..."):
                        create_mcq_questions(documents)
                        # display_mcq_questions(mcq_questions)

            with tabs[2]:
                if st.button("Create Notes"):
                    with st.spinner("Creating notes..."):
                        placeholder = st.empty()
                        create_notes(documents, placeholder)

            
            st.subheader("Chat with the PDF")
            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                relevant_docs = retrieve_documents(user_question,documents)
                context = " ".join(relevant_docs)
                prompt = f"""Based on the following context from the PDF, please answer the question.

                Context: {context}

                Question: {user_question}

                Answer:"""
                with st.spinner("Generating answer..."):
                    placeholder = st.empty()
                    stream_response(prompt, placeholder)

    else:
        st.info("Please upload a PDF file to begin.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
