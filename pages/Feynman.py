# import streamlit as st
# import PyPDF2 
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from openai import OpenAI
# import io

# st.set_page_config(page_title="PDF Analysis with Llama 2", page_icon="📚")

# # Point to the local server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# # ... (keep all the other functions unchanged)

# def encode_texts(query, documents):
#     # Combine the query and documents to build a consistent vocabulary
#     corpus = [query] + documents
    
#     # Initialize and fit the vectorizer on the entire corpus
#     vectorizer = TfidfVectorizer()
#     vectorizer.fit(corpus)
    
#     # Transform the query and documents using the same vectorizer
#     query_embedding = vectorizer.transform([query]).toarray()
#     doc_embeddings = vectorizer.transform(documents).toarray()
    
#     return query_embedding, doc_embeddings

# def retrieve_documents(query, documents, top_k=3):
#     # Encode the query and documents to ensure consistent dimensions
#     query_embedding, doc_embeddings = encode_texts(query, documents)

#     # Debugging: Print the shapes of the embeddings
#     print(f"Query Embedding Shape: {query_embedding.shape}")
#     print(f"Document Embedding Shape: {doc_embeddings.shape}")

#     # Calculate cosine similarities
#     similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

#     # Retrieve the top k most similar documents
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [documents[i] for i in top_indices]

# def stream_response(prompt, placeholder):
#     response = ""
#     for chunk in generate_llama_response(prompt):
#         if chunk.choices[0].delta.content:
#             response += chunk.choices[0].delta.content
#             placeholder.markdown(response + "▌")
#     placeholder.markdown(response)
#     return response

# def summarize_pdf(documents, placeholder):
#     context = " ".join(documents[:3])
#     prompt = f"""Provide a concise summary of the main points in the following text:

# Text to summarize: {context}

# Summary:"""
#     return stream_response(prompt, placeholder)

# def generate_llama_response(prompt, stream=True):
#     messages = [
#         {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
#         {"role": "user", "content": prompt}
#     ]
    
#     completion = client.chat.completions.create(
#         model="TheBloke/Llama-2-7B-Chat-GGUF",
#         messages=messages,
#         temperature=0.7,
#         stream=stream  # Pass the stream parameter to the API call
#     )
    
#     if stream:
#         return completion
#     else:
#         try:
#             return completion.choices[0].message.content
#         except AttributeError:
#             # If the structure is different, try accessing content directly
#             return completion.choices[0].content
# def create_mcq_questions(documents):
#     context = " ".join(documents[:3])
#     prompt = f"""Generate 2 multiple-choice questions based on the following text. Each question should have 4 options (A, B, C, D) with one correct answer. Format your response EXACTLY as follows for each question:

# Question [number]: [Question text]
# Options: A) [Option A] B) [Option B] C) [Option C] D) [Option D]
# Correct Answer: [Full text of correct answer]

# Each question should start be on a new line

# Text to create questions from: {context}

# Questions:"""
   
#     response = generate_llama_response(prompt, False)
#     return parse_mcq_questions(response)  
 
# def parse_mcq_questions(response):
#     print(response)
#     try:
#         questions = []
#         quest_answr = {}
#         n = 1
        
#         # Split response into lines and process each line
#         for line in response.split('\n'):
#             line = line.strip()
#             # Skip empty lines
#             if not line:
#                 continue
            
#             # Process question
#             if line.startswith(f'Question {n}:'):
#                 if quest_answr:
#                     questions.append(quest_answr)
#                 quest_answr = {"question": line.split(f'Question {n}: ')[1], "options": {}, "correct_answer": ""}
#                 n += 1
            
#             # Process options
#             elif line.startswith('Options:'):
#                 try:
#                     options = line.split('Options: ')[1].split(' ')
#                     # Initialize options dictionary
#                     quest_answr['options'] = {
#             "A": " ".join(options[1:options.index('B)')]),
#             "B": " ".join(options[options.index('B)') + 1:options.index('C)')]),
#             "C": " ".join(options[options.index('C)') + 1:options.index('D)')]),
#             "D": " ".join(options[options.index('D)') + 1:])
#         }
#                 except Exception as e:
#                     st.error(f"Error processing options: {str(e)}")
#                     continue
            
#             # Process correct answer
#             elif line.startswith('Correct Answer:'):
#                 try:
#                     correct_ans = line.split('Correct Answer: ')[1]
#                     correct_ans_letter = correct_ans[0]
                    
#                     # Verify the correct answer letter exists in options
#                     if correct_ans_letter in quest_answr['options']:
#                         quest_answr["correct_answer"] = quest_answr["options"][correct_ans_letter]
#                     else:
#                         # If letter not found, store the full correct answer text
#                         quest_answr["correct_answer"] = correct_ans.strip()
#                 except Exception as e:
#                     st.error(f"Error processing correct answer: {str(e)}")
#                     continue
        
#         # Append the last question if it exists and has required fields
#         if quest_answr and 'question' in quest_answr and quest_answr['options']:
#             questions.append(quest_answr)
#         return questions
#     except Exception as e:
#         st.error(f"Error parsing questions: {str(e)}")
#         return []

# def display_mcq_questions(questions):
#     if not questions:
#         st.warning("No questions to display")
#         return
    
#     # Initialize answer storage in session state if not exists
#     if "answers" not in st.session_state:
#         st.session_state.answers = {}
#     if "show_results" not in st.session_state:
#         st.session_state.show_results = False
    
#     # Display questions and collect answers
#     for i, q in enumerate(questions):
#         st.write(f"Question {i + 1}: {q['question']}")
        
#         # Create radio button with stored answer if it exists
#         answer_key = f"answer_{i}"
#         options = list(q['options'].values()) + ["Clear response"]
        
#         # Get the index of the previously selected option
#         default_index = len(options) - 1  # Default to "Clear response"
#         if answer_key in st.session_state.answers:
#             try:
#                 default_index = options.index(st.session_state.answers[answer_key])
#             except ValueError:
#                 default_index = len(options) - 1

#         selected_option = st.radio(
#             'Choose an option:',
#             options,
#             key=f"radio_{i}",
#             index=default_index
#         )
        
#         # Store the answer in session state
#         if selected_option != "Clear response":
#             st.session_state.answers[answer_key] = selected_option
#         elif answer_key in st.session_state.answers:
#             del st.session_state.answers[answer_key]
            
#         st.write('---')

#     # Submit button
#     if st.button('Submit'):
#         st.session_state.show_results = True
    
#     # Show results if submit was clicked
#     if st.session_state.show_results:
#         total_correct = 0
#         questions_attempted = 0
        
#         for i, q in enumerate(questions):
#             answer_key = f"answer_{i}"
#             selected_answer = st.session_state.answers.get(answer_key)
            
#             if selected_answer and selected_answer != "Clear response":
#                 questions_attempted += 1
#                 if selected_answer == q["correct_answer"]:
#                     st.success(f"Question {i+1}: Correct!")
#                     total_correct += 1
#                 else:
#                     st.error(f"Question {i+1}: Wrong!")
#                 st.write(f"The correct answer is: {q['correct_answer']}")
#             else:
#                 st.warning(f"Question {i+1}: No answer selected")
        
#         if questions_attempted > 0:
#             score_percentage = (total_correct / len(questions)) * 100
#             st.write(f"\nFinal Score: {total_correct}/{len(questions)} ({score_percentage:.1f}%)")




# def create_notes(documents, placeholder):
#     context = " ".join(documents[:3])
#     prompt = f"""Create concise and informative notes based on the following text. Focus on key concepts, definitions, and important points:

# Text to create notes from: {context}

# Notes:"""
#     return stream_response(prompt, placeholder)
# # import PyPDF2

# def extract_text_from_pdf(file):
#     pdf_reader = PyPDF2.PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Streamlit App
# def main():
#     st.title("PDF Analysis with Llama 2")
  
#     uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
#     if uploaded_file is not None:
#             text = extract_text_from_pdf(uploaded_file)
#             chunk_size = 1000
#             documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#             st.success("PDF uploaded and processed successfully!")

#             tabs = st.tabs(["Summarize PDF", "Generate MCQ Questions", "Create Notes"])
            
#             if "load_state" not in st.session_state:
#                 st.session_state.load_state = False
#             if "generation_complete" not in st.session_state:
#                 st.session_state.generation_complete = False

#             with tabs[0]:
#                 if st.button("Summarize PDF"):
#                     with st.spinner("Generating summary..."):
#                         placeholder = st.empty()
#                         summarize_pdf(documents, placeholder)

#             with tabs[1]:
#                 # Only generate questions when the button is clicked and questions haven't been generated yet
#                 if st.button("Generate MCQ Questions") and not st.session_state.generation_complete:
#                     with st.spinner("Generating MCQ questions..."):
#                         st.session_state.questions = create_mcq_questions(documents)
#                         st.session_state.generation_complete = True
#                         st.session_state.load_state = True

#                 # Display questions if they exist in session state
#                 if st.session_state.generation_complete and st.session_state.questions:
#                     display_mcq_questions(st.session_state.questions)

#                     # Add a reset button if you want to allow regenerating questions
#                     if st.button("Reset Questions"):
#                         st.session_state.generation_complete = False
#                         st.session_state.questions = None
#                         st.session_state.load_state = False
#                         st.experimental_rerun()

#             with tabs[2]:
#                 if st.button("Create Notes"):
#                     with st.spinner("Creating notes..."):
#                         placeholder = st.empty()
#                         create_notes(documents, placeholder)

#             st.subheader("Chat with the PDF")
#             user_question = st.text_input("Ask a question about the PDF:")
#             if user_question:
#                 relevant_docs = retrieve_documents(user_question,documents)
#                 context = " ".join(relevant_docs)
#                 prompt = f"""Based on the following context from the PDF, please answer the question.

#                 Context: {context}

#                 Question: {user_question}

#                 Answer:"""
#                 with st.spinner("Generating answer..."):
#                     placeholder = st.empty()
#                     stream_response(prompt, placeholder)

#     else:
#         st.info("Please upload a PDF file to begin.")

# if __name__ == "__main__":
#     main()








# ====================================================================




import streamlit as st
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import time
from datetime import datetime

# Initialize OpenAI client for local LLM
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'load_state' not in st.session_state:
    st.session_state.load_state = False
if 'study_history' not in st.session_state:
    st.session_state.study_history = []
if 'current_pdf_text' not in st.session_state:
    st.session_state.current_pdf_text = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

# [Previous helper functions remain the same: extract_text_from_pdf, encode_texts, 
# retrieve_documents, stream_response, summarize_pdf, create_notes, create_mcq_questions, 
# parse_mcq_questions, display_mcq_questions]
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

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
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
    for chunk in generate_llama_response(prompt):
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

def generate_llama_response(prompt, stream=True):
    messages = [
        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
        {"role": "user", "content": prompt}
    ]
    
    completion = client.chat.completions.create(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        messages=messages,
        temperature=0.7,
        stream=stream  # Pass the stream parameter to the API call
    )
    
    if stream:
        return completion
    else:
        try:
            return completion.choices[0].message.content
        except AttributeError:
            # If the structure is different, try accessing content directly
            return completion.choices[0].content
def create_mcq_questions(documents):
    context = " ".join(documents[:3])
    prompt = f"""Generate 2 multiple-choice questions based on the following text. Each question should have 4 options (A, B, C, D) with one correct answer. Format your response EXACTLY as follows for each question:

Question [number]: [Question text]
Options: A) [Option A] B) [Option B] C) [Option C] D) [Option D]
Correct Answer: [Full text of correct answer]

Each question should start be on a new line

Text to create questions from: {context}

Questions:"""
   
    response = generate_llama_response(prompt, False)
    return parse_mcq_questions(response)  
 
def parse_mcq_questions(response):
    try:
        questions = []
        quest_answr = {}
        n = 1
        
        # Split response into lines and process each line
        for line in response.split('\n'):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            
            # Process question
            if line.startswith(f'Question {n}:'):
                if quest_answr:
                    questions.append(quest_answr)
                quest_answr = {"question": line.split(f'Question {n}: ')[1], "options": {}, "correct_answer": ""}
                n += 1
            
            # Process options
            elif line.startswith('Options:'):
                try:
                    options = line.split('Options: ')[1].split(' ')
                    # Initialize options dictionary
                    quest_answr['options'] = {
            "A": " ".join(options[1:options.index('B)')]),
            "B": " ".join(options[options.index('B)') + 1:options.index('C)')]),
            "C": " ".join(options[options.index('C)') + 1:options.index('D)')]),
            "D": " ".join(options[options.index('D)') + 1:])
        }
                except Exception as e:
                    st.error(f"Error processing options: {str(e)}")
                    continue
            
            # Process correct answer
            elif line.startswith('Correct Answer:'):
                try:
                    correct_ans = line.split('Correct Answer: ')[1]
                    correct_ans_letter = correct_ans[0]
                    
                    # Verify the correct answer letter exists in options
                    if correct_ans_letter in quest_answr['options']:
                        quest_answr["correct_answer"] = quest_answr["options"][correct_ans_letter]
                    else:
                        # If letter not found, store the full correct answer text
                        quest_answr["correct_answer"] = correct_ans.strip()
                except Exception as e:
                    st.error(f"Error processing correct answer: {str(e)}")
                    continue
        
        # Append the last question if it exists and has required fields
        if quest_answr and 'question' in quest_answr and quest_answr['options']:
            questions.append(quest_answr)
        return questions
    except Exception as e:
        st.error(f"Error parsing questions: {str(e)}")
        return []

def display_mcq_questions(questions):
    if not questions:
        st.warning("No questions to display")
        return
    
    # Initialize answer storage in session state if not exists
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Display questions and collect answers
    for i, q in enumerate(questions):
        st.write(f"Question {i + 1}: {q['question']}")
        
        # Create radio button with stored answer if it exists
        answer_key = f"answer_{i}"
        options = list(q['options'].values()) + ["Clear response"]
        
        # Get the index of the previously selected option
        default_index = len(options) - 1  # Default to "Clear response"
        if answer_key in st.session_state.answers:
            try:
                default_index = options.index(st.session_state.answers[answer_key])
            except ValueError:
                default_index = len(options) - 1

        selected_option = st.radio(
            'Choose an option:',
            options,
            key=f"radio_{i}",
            index=default_index
        )
        
        # Store the answer in session state
        if selected_option != "Clear response":
            st.session_state.answers[answer_key] = selected_option
        elif answer_key in st.session_state.answers:
            del st.session_state.answers[answer_key]
            
        st.write('---')

    # Submit button
    if st.button('Submit'):
        st.session_state.show_results = True
    
    # Show results if submit was clicked
    if st.session_state.show_results:
        total_correct = 0
        questions_attempted = 0
        
        for i, q in enumerate(questions):
            answer_key = f"answer_{i}"
            selected_answer = st.session_state.answers.get(answer_key)
            
            if selected_answer and selected_answer != "Clear response":
                questions_attempted += 1
                if selected_answer == q["correct_answer"]:
                    st.success(f"Question {i+1}: Correct!")
                    total_correct += 1
                else:
                    st.error(f"Question {i+1}: Wrong!")
                st.write(f"The correct answer is: {q['correct_answer']}")
            else:
                st.warning(f"Question {i+1}: No answer selected")
        
        if questions_attempted > 0:
            score_percentage = (total_correct / len(questions)) * 100
            st.write(f"\nFinal Score: {total_correct}/{len(questions)} ({score_percentage:.1f}%)")




def create_notes(documents, placeholder):
    context = " ".join(documents[:3])
    prompt = f"""Create concise and informative notes based on the following text. Focus on key concepts, definitions, and important points:

Text to create notes from: {context}

Notes:"""
    return stream_response(prompt, placeholder)

def start_timer():
    """Start the study session timer"""
    st.session_state.start_time = time.time()
    st.session_state.timer_running = True

def end_timer():
    """End the study session timer and return duration"""
    if st.session_state.start_time is not None:
        end_time = time.time()
        duration = round((end_time - st.session_state.start_time) / 60)
        st.session_state.start_time = None
        st.session_state.timer_running = False
        return duration
    return 0

def save_study_session(pdf_name, duration):
    """Save study session details"""
    st.session_state.study_history.append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'pdf': pdf_name,
        'duration': duration
    })

def main():
    st.set_page_config(page_title="Feynman Technique PDF Analysis", page_icon="📚", layout="wide")
    
    # Sidebar content
    with st.sidebar:
        st.markdown("""
        <h2 style='text-align: center;'>
            📚 Document Control
        </h2>
        """, unsafe_allow_html=True)
        
        # File uploader in sidebar
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None and (st.session_state.current_file_name != uploaded_file.name):
            st.session_state.current_file_name = uploaded_file.name
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.current_pdf_text = text
            chunk_size = 1000
            st.session_state.documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            st.success("PDF uploaded and processed successfully!")
        
        st.divider()
        
        # Timer controls
        if st.session_state.current_pdf_text is not None:
            st.markdown("### ⏱️ Study Session Control")
            if not st.session_state.timer_running:
                if st.button("Start Study Session", type="primary"):
                    start_timer()
                    st.success("Study session started!")
            else:
                if st.button("End Study Session", type="secondary"):
                    duration = end_timer()
                    save_study_session(st.session_state.current_file_name, duration)
                    st.success(f"Study session recorded: {duration} minutes")
            
            if st.session_state.timer_running:
                st.info("📝 Study session in progress...")
        
        st.divider()
        
        # Study history
        st.markdown("### 📊 Study Statistics")
        if st.session_state.study_history:
            for session in st.session_state.study_history:
                st.markdown(f"""
                📅 {session['date']}
                - Document: {session['pdf']}
                - Duration: {session['duration']} minutes
                """)
        else:
            st.info("No study sessions recorded yet")
    
    # Main content area
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        📚 Feynman Technique PDF Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_pdf_text is None:
        st.info("👈 Please upload a PDF file in the sidebar to begin your study session.")
        st.divider()
        st.markdown("""
        ### 🎯 Features Available:
        - 📝 Comprehensive PDF Summary
        - 📒 Detailed Study Notes Generation
        - ❓ Interactive MCQ Questions
        - 💬 Chat with your PDF
        - ⏱️ Study Session Tracking
        """)
    else:
        # Study content tabs
        tabs = st.tabs(["📝 Summary", "📒 Note Making", "❓ MCQ Questions", "💬 Chat with PDF"])
        
        with tabs[0]:
            if st.button("Generate Summary"):
                with st.spinner("Applying Feynman Technique..."):
                    placeholder = st.empty()
                    summarize_pdf(st.session_state.documents, placeholder)
        
        with tabs[1]:
            if st.button("Create Study Notes"):
                with st.spinner("Creating detailed notes..."):
                    placeholder = st.empty()
                    create_notes(st.session_state.documents, placeholder)
        
        with tabs[2]:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Generate New Questions") and not st.session_state.generation_complete:
                    with st.spinner("Creating questions..."):
                        st.session_state.questions = create_mcq_questions(st.session_state.documents)
                        st.session_state.generation_complete = True
                        st.session_state.load_state = True
            
            with col2:
                if st.button("Reset Questions"):
                    st.session_state.generation_complete = False
                    st.session_state.questions = None
                    st.session_state.load_state = False
                    st.experimental_rerun()
            
            if st.session_state.generation_complete and st.session_state.questions:
                display_mcq_questions(st.session_state.questions)
        
        with tabs[3]:
            st.markdown("""
            ### Chat with your PDF
            Ask questions about the content and get explanations using the Feynman Technique
            """)
            
            user_question = st.text_input("What would you like to understand better?")
            
            if user_question:
                relevant_docs = retrieve_documents(user_question, st.session_state.documents)
                context = " ".join(relevant_docs)
                prompt = f"""Using the Feynman Technique, please explain this concept in simple terms:
                
                Context: {context}
                Question: {user_question}
                
                Explanation:"""
                
                with st.spinner("Crafting explanation..."):
                    placeholder = st.empty()
                    stream_response(prompt, placeholder)

if __name__ == "__main__":
    main()






    