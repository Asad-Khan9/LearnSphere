import streamlit as st
import PyPDF2 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import io

st.set_page_config(page_title="PDF Analysis with Llama 2", page_icon="📚")

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ... (keep all the other functions unchanged)

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
# import PyPDF2
def create_flashcards(documents, num_cards=5):
    """
    Generate flash cards from PDF content using LLaMA model.
    
    Args:
        documents (list): List of text chunks from the PDF
        num_cards (int): Number of flash cards to generate
        
    Returns:
        list: List of dictionaries containing flash cards
    """
    context = " ".join(documents[:3])
    prompt = f"""Create {num_cards} flash cards from the following text. Each flash card should include:
    1. A key term or concept
    2. A brief one-sentence definition or explanation
    3. An example or additional context (optional)

Format each flash card exactly as follows:
Term: [term]
Definition: [one-sentence definition]
Example: [brief example or context]

Text to create flash cards from: {context}

Flash cards:"""

    try:
        response = generate_llama_response(prompt, stream=False)
        return parse_flashcards(response)
    except Exception as e:
        st.error(f"Error generating flash cards: {str(e)}")
        return []

def parse_flashcards(response):
    """
    Parse the LLaMA response into structured flash cards.
    
    Args:
        response (str): Raw response from LLaMA
        
    Returns:
        list: List of dictionaries containing parsed flash cards
    """
    flashcards = []
    current_card = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            if current_card:
                flashcards.append(current_card)
                current_card = {}
            continue
            
        if line.startswith('Term:'):
            if current_card:
                flashcards.append(current_card)
            current_card = {'term': line[5:].strip()}
        elif line.startswith('Definition:'):
            current_card['definition'] = line[11:].strip()
        elif line.startswith('Example:'):
            current_card['example'] = line[8:].strip()
            
    if current_card:
        flashcards.append(current_card)
        
    return flashcards

def display_flashcards(flashcards):
    """
    Display flash cards in the Streamlit interface with flip animation.
    
    Args:
        flashcards (list): List of flash card dictionaries
    """
    if not flashcards:
        st.warning("No flash cards to display")
        return
        
    # Add CSS for flip animation
    st.markdown("""
        <style>
        .card {
            position: relative;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 1rem;
            background-color: white;
        }
        .term {
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
        .definition {
            color: #666;
            margin-bottom: 0.5rem;
        }
        .example {
            font-style: italic;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for tracking flipped cards
    if 'flipped_cards' not in st.session_state:
        st.session_state.flipped_cards = [False] * len(flashcards)
    
    for i, card in enumerate(flashcards):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
                <div class="card">
                    <div class="term">{card['term']}</div>
                    {'<div class="definition">' + card['definition'] + '</div>' if st.session_state.flipped_cards[i] else ''}
                    {'<div class="example">' + card.get('example', '') + '</div>' if st.session_state.flipped_cards[i] and 'example' in card else ''}
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            if st.button('Flip' if not st.session_state.flipped_cards[i] else 'Hide', key=f'flip_{i}'):
                st.session_state.flipped_cards[i] = not st.session_state.flipped_cards[i]
                st.rerun()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit App
def main():
    st.title("PDF Analysis with Llama 2")
    with st.sidebar:

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        st.success("PDF uploaded and processed successfully!")


    if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            chunk_size = 1000
            documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


            tabs = st.tabs(["Summarize PDF", "Generate MCQ Questions", "Create Notes","Flash Card"])
            
            if "load_state" not in st.session_state:
                st.session_state.load_state = False
            if "generation_complete" not in st.session_state:
                st.session_state.generation_complete = False

            with tabs[0]:
                if st.button("Summarize PDF"):
                    with st.spinner("Generating summary..."):
                        placeholder = st.empty()
                        summarize_pdf(documents, placeholder)

            with tabs[1]:
                # Only generate questions when the button is clicked and questions haven't been generated yet
                if st.button("Generate MCQ Questions") and not st.session_state.generation_complete:
                    with st.spinner("Generating MCQ questions..."):
                        st.session_state.questions = create_mcq_questions(documents)
                        st.session_state.generation_complete = True
                        st.session_state.load_state = True

                # Display questions if they exist in session state
                if st.session_state.generation_complete and st.session_state.questions:
                    display_mcq_questions(st.session_state.questions)

                    # Add a reset button if you want to allow regenerating questions
                    if st.button("Reset Questions"):
                        st.session_state.generation_complete = False
                        st.session_state.questions = None
                        st.session_state.load_state = False
                        st.rerun()

            with tabs[2]:
                if st.button("Create Notes"):
                    with st.spinner("Creating notes..."):
                        placeholder = st.empty()
                        create_notes(documents, placeholder)
            with tabs[3]:  # New Flash Cards tab
                col1, col2 = st.columns([3, 1])
                with col1:
                    num_cards = st.slider("Number of flash cards to generate:", 3, 10, 5)
                with col2:
                    generate_button = st.button("Generate Flash Cards")

                if generate_button:
                    with st.spinner("Generating flash cards..."):
                        cards = create_flashcards(documents, num_cards)
                        st.session_state.flashcards = cards

                if 'flashcards' in st.session_state:
                    display_flashcards(st.session_state.flashcards)

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

if __name__ == "__main__":
    main()






# =====================================================================================



# import streamlit as st
# import PyPDF2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from openai import OpenAI
# import time
# from datetime import datetime, timedelta

# import random
# # Initialize OpenAI client for local LLM
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# # Initialize session state variables
# if 'questions' not in st.session_state:
#     st.session_state.questions = None
# if 'generation_complete' not in st.session_state:
#     st.session_state.generation_complete = False
# if 'flashcards' not in st.session_state:
#     st.session_state.flashcards = []
# if 'current_pdf_text' not in st.session_state:
#     st.session_state.current_pdf_text = None
# if 'study_history' not in st.session_state:
#     st.session_state.study_history = []
# if 'flashcard_boxes' not in st.session_state:
#     # Leitner system boxes with review intervals
#     st.session_state.flashcard_boxes = {
#         1: {'cards': [], 'review_interval': timedelta(days=1)},    # Daily
#         2: {'cards': [], 'review_interval': timedelta(days=3)},    # Every 3 days
#         3: {'cards': [], 'review_interval': timedelta(days=7)},    # Weekly
#         4: {'cards': [], 'review_interval': timedelta(days=14)},   # Biweekly
#         5: {'cards': [], 'review_interval': timedelta(days=30)}    # Monthly
#     }
# if 'last_review_date' not in st.session_state:
#     st.session_state.last_review_date = {}

# def extract_text_from_pdf(file):
#     """Extract text from uploaded PDF file"""
#     pdf_reader = PyPDF2.PdfReader(file)
#     text = ""
#     with st.spinner("Extracting text from PDF..."):
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def create_flashcards(documents, num_cards):
#     """Generate flashcards using LLM"""
#     prompt = f"""Create {num_cards} flash cards from the following text. 
#     Format each card as JSON with 'front' and 'back' fields. 
#     Make the cards follow increasing difficulty levels.
    
#     Text: {' '.join(documents[:3])}
    
#     Format each card as:
#     {{
#         "front": "question/concept",
#         "back": "answer/explanation",
#         "difficulty": "1-5"
#     }}"""
    
#     response = client.chat.completions.create(
#         model="TheBloke/Llama-2-7B-Chat-GGUF",
#         messages=[
#             {"role": "system", "content": "You are an expert at creating educational flashcards."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7,
#         stream=False
#     )
    
#     # Parse the response into structured flashcard data
#     import json
#     try:
#         cards = json.loads(response.choices[0].message.content)
#         for card in cards:
#             card['box'] = 1  # Start all cards in box 1
#             card['next_review'] = datetime.now()
#         return cards
#     except:
#         st.error("Error parsing flashcards. Please try again.")
#         return []

# def move_card_to_box(card_index, current_box, correct):
#     """Move card between Leitner boxes based on performance"""
#     card = st.session_state.flashcard_boxes[current_box]['cards'][card_index]
    
#     # Remove card from current box
#     st.session_state.flashcard_boxes[current_box]['cards'].pop(card_index)
    
#     # Move to next box if correct, previous box if wrong
#     new_box = min(5, current_box + 1) if correct else max(1, current_box - 1)
    
#     # Update next review date based on box interval
#     card['next_review'] = datetime.now() + st.session_state.flashcard_boxes[new_box]['review_interval']
    
#     # Add to new box
#     st.session_state.flashcard_boxes[new_box]['cards'].append(card)
    
#     return new_box

# def display_flashcards(cards):
#     """Display flashcards with Leitner system integration"""
#     st.markdown("### 📝 Flashcards Review")
    
#     # Display box statistics
#     cols = st.columns(5)
#     for i, col in enumerate(cols, 1):
#         col.metric(f"Box {i}", len(st.session_state.flashcard_boxes[i]['cards']))
    
#     # Show cards due for review
#     today = datetime.now()
#     due_cards = []
#     for box_num, box in st.session_state.flashcard_boxes.items():
#         for i, card in enumerate(box['cards']):
#             if card['next_review'] <= today:
#                 due_cards.append((box_num, i, card))
    
#     if due_cards:
#         st.markdown(f"### Cards Due for Review: {len(due_cards)}")
#         for box_num, card_index, card in due_cards:
#             with st.expander(f"Card from Box {box_num}"):
#                 st.markdown(f"**Question:** {card['front']}")
#                 if st.button("Show Answer", key=f"show_{box_num}_{card_index}"):
#                     st.markdown(f"**Answer:** {card['back']}")
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if st.button("I knew this ✅", key=f"correct_{box_num}_{card_index}"):
#                             new_box = move_card_to_box(card_index, box_num, True)
#                             st.success(f"Card moved to Box {new_box}")
#                             st.experimental_rerun()
#                     with col2:
#                         if st.button("Need more practice ❌", key=f"wrong_{box_num}_{card_index}"):
#                             new_box = move_card_to_box(card_index, box_num, False)
#                             st.info(f"Card moved to Box {new_box}")
#                             st.experimental_rerun()
#     else:
#         st.success("No cards due for review! Come back later.")

# def create_mcq_questions(documents):
#     """Generate multiple choice questions"""
#     prompt = """Create 5 multiple choice questions based on this text. Format as:
#     Q[number]: [Question]
#     A) [Option A]
#     B) [Option B]
#     C) [Option C]
#     D) [Option D]
#     Correct: [A/B/C/D]
#     Explanation: [Brief explanation]
    
#     Text: {}"""
    
#     response = client.chat.completions.create(
#         model="TheBloke/Llama-2-7B-Chat-GGUF",
#         messages=[
#             {"role": "system", "content": "You are an expert at creating educational assessments."},
#             {"role": "user", "content": prompt.format(" ".join(documents[:3]))}
#         ],
#         temperature=0.7,
#         stream=False
#     )
#     return response.choices[0].message.content

# def parse_mcq_questions(questions_text):
#     """Parse MCQ questions into structured format"""
#     questions = []
#     current_question = {}
    
#     for line in questions_text.split('\n'):
#         line = line.strip()
#         if line.startswith('Q'):
#             if current_question:
#                 questions.append(current_question)
#             current_question = {'question': line[line.find(':')+1:].strip()}
#         elif line.startswith(('A)', 'B)', 'C)', 'D)')):
#             option = line[0]
#             current_question[f'option_{option}'] = line[2:].strip()
#         elif line.startswith('Correct:'):
#             current_question['correct'] = line[8:].strip()
#         elif line.startswith('Explanation:'):
#             current_question['explanation'] = line[12:].strip()
    
#     if current_question:
#         questions.append(current_question)
#     return questions

# def display_mcq_questions(questions_text):
#     """Display MCQ questions with interactive elements"""
#     questions = parse_mcq_questions(questions_text)
    
#     for i, q in enumerate(questions, 1):
#         st.subheader(f"Question {i}")
#         st.write(q['question'])
        
#         options = {
#             'A': q.get('option_A', ''),
#             'B': q.get('option_B', ''),
#             'C': q.get('option_C', ''),
#             'D': q.get('option_D', '')
#         }
        
#         answer = st.radio("Select your answer:", options.keys(), key=f"q_{i}")
        
#         if st.button("Check Answer", key=f"check_{i}"):
#             if answer == q['correct']:
#                 st.success("Correct! 🎉")
#             else:
#                 st.error(f"Incorrect. The correct answer is {q['correct']}")
            
#             with st.expander("See Explanation"):
#                 st.write(q['explanation'])
        
#         st.divider()

# def summarize_pdf(documents, placeholder):
#     """Generate a comprehensive summary"""
#     prompt = """Create a clear summary of the following text, organizing key concepts
#     from basic to advanced:
    
#     Text: {}"""
    
#     messages = [
#         {"role": "system", "content": "You are an expert at creating educational summaries."},
#         {"role": "user", "content": prompt.format(" ".join(documents[:3]))}
#     ]
    
#     response = client.chat.completions.create(
#         model="TheBloke/Llama-2-7B-Chat-GGUF",
#         messages=messages,
#         temperature=0.7,
#         stream=True
#     )
    
#     full_response = ""
#     for chunk in response:
#         if chunk.choices[0].delta.content is not None:
#             full_response += chunk.choices[0].delta.content
#             placeholder.markdown(full_response + "▌")
#     placeholder.markdown(full_response)

# def create_notes(documents, placeholder):
#     """Generate structured study notes"""
#     prompt = """Create detailed study notes from this material. Include:
#     1. Key Concepts
#     2. Important Definitions
#     3. Examples
#     4. Common Misconceptions
#     5. Practice Problems
    
#     Text: {}"""
    
#     messages = [
#         {"role": "system", "content": "You are an expert at creating educational notes."},
#         {"role": "user", "content": prompt.format(" ".join(documents[:3]))}
#     ]
    
#     response = client.chat.completions.create(
#         model="TheBloke/Llama-2-7B-Chat-GGUF",
#         messages=messages,
#         temperature=0.7,
#         stream=True
#     )
    
#     full_response = ""
#     for chunk in response:
#         if chunk.choices[0].delta.content is not None:
#             full_response += chunk.choices[0].delta.content
#             placeholder.markdown(full_response + "▌")
#     placeholder.markdown(full_response)

# def main():
#     st.set_page_config(page_title="Leitner Technique PDF Analysis", page_icon="📚", layout="wide")
    
#     # Main title with styling
#     st.markdown("""
#     <h1 style='text-align: center; color: #1f77b4;'>
#         📚 Leitner Technique PDF Analysis
#     </h1>
#     """, unsafe_allow_html=True)
    
#     # Sidebar with Leitner System explanation and statistics
#     with st.sidebar:
#         st.header("📊 Leitner System Statistics")
#         # Main content area
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         # Show box statistics
        
#         # Show study tips
#         with st.expander("ℹ️ About Leitner System"):
#             st.markdown("""
#             The Leitner System is a method of spaced repetition:
#             - Cards start in Box 1 (daily review)
#             - Correct answers move up a box
#             - Incorrect answers move back to Box 1
#             - Higher boxes have longer review intervals
#             """)
    
#     # for box_num, box in st.session_state.flashcard_boxes.items():
#     #     st.metric(
#     #         f"Box {box_num}",
#     #         len(box['cards']),
#     #         f"Review every {box['review_interval'].days} days"
#     #     )
#     # Create columns - one for each box
#     cols = st.columns(len(st.session_state.flashcard_boxes))

#     # Display metrics in each column
#     for (box_num, box), col in zip(st.session_state.flashcard_boxes.items(), cols):
#         with col:
#             st.metric(
#                 f"Box {box_num}",
#                 len(box['cards']),
#                 f"Review every {box['review_interval'].days} days"
#             )
    
#     if uploaded_file is not None:
#         if st.session_state.current_pdf_text is None:
#             text = extract_text_from_pdf(uploaded_file)
#             st.session_state.current_pdf_text = text
#             chunk_size = 1000
#             documents = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#             st.success("PDF uploaded and processed successfully!")
#         else:
#             documents = [st.session_state.current_pdf_text[i:i+1000] 
#                         for i in range(0, len(st.session_state.current_pdf_text), 1000)]
        
#         # Study content tabs
#         tabs = st.tabs(["📝 Summary", "📒 Notes", "❓ MCQ", "🎴 Flash Cards"])
        
#         with tabs[0]:
#             if st.button("Generate Summary"):
#                 with st.spinner("Creating summary..."):
#                     placeholder = st.empty()
#                     summarize_pdf(documents, placeholder)
        
#         with tabs[1]:
#             if st.button("Create Study Notes"):
#                 with st.spinner("Creating notes..."):
#                     placeholder = st.empty()
#                     create_notes(documents, placeholder)
        
#         with tabs[2]:
#             col1, col2 = st.columns([3, 1])
#             with col1:
#                 if st.button("Generate New Questions") and not st.session_state.generation_complete:
#                     with st.spinner("Creating questions..."):
#                         st.session_state.questions = create_mcq_questions(documents)
#                         st.session_state.generation_complete = True
            
#             with col2:
#                 if st.button("Reset Questions"):
#                     st.session_state.generation_complete = False
#                     st.session_state.questions = None
#                     st.experimental_rerun()
            
#             if st.session_state.generation_complete and st.session_state.questions:
#                 display_mcq_questions(st.session_state.questions)
        
#         with tabs[3]:
#             st.markdown("### 🎴 Flash Cards")
            
#             col1, col2, col3 = st.columns([2, 1, 1])
#             with col1:
#                 num_cards = st.slider("Number of flash cards to generate:", 3, 10, 5)
#             with col2:
#                 if st.button("Generate New Cards"):
#                     with st.spinner("Creating flash cards..."):
#                         cards = create_flashcards(documents, num_cards)
#                         # Add new cards to Box 1
#                         st.session_state.flashcard_boxes[1]['cards'].extend(cards)
#                         st.success(f"Generated {len(cards)} new cards!")
#             with col3:
#                 if st.button("Review Due Cards"):
#                     st.session_state.reviewing = True
            
#             if getattr(st.session_state, 'reviewing', False):
#                 due_cards = []
#                 current_date = datetime.datetime.now()
                
#                 # Collect due cards from all boxes
#                 for box_num, box in st.session_state.flashcard_boxes.items():
#                     for card in box['cards']:
#                         if 'last_reviewed' not in card or \
#                            current_date - card['last_reviewed'] >= box['review_interval']:
#                             card['current_box'] = box_num
#                             due_cards.append(card)
                
#                 if due_cards:
#                     card = random.choice(due_cards)
#                     st.markdown(f"**Question:**\n\n{card['question']}")
                    
#                     if st.button("Show Answer"):
#                         st.markdown(f"**Answer:**\n\n{card['answer']}")
                        
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             if st.button("Correct ✅"):
#                                 # Move to next box if not in highest box
#                                 current_box = card['current_box']
#                                 if current_box < 5:  # 5 is the highest box
#                                     st.session_state.flashcard_boxes[current_box]['cards'].remove(card)
#                                     st.session_state.flashcard_boxes[current_box + 1]['cards'].append(card)
#                                 card['last_reviewed'] = current_date
#                                 st.experimental_rerun()
                        
#                         with col2:
#                             if st.button("Incorrect ❌"):
#                                 # Move back to box 1
#                                 current_box = card['current_box']
#                                 if current_box > 1:
#                                     st.session_state.flashcard_boxes[current_box]['cards'].remove(card)
#                                     st.session_state.flashcard_boxes[1]['cards'].append(card)
#                                 card['last_reviewed'] = current_date
#                                 st.experimental_rerun()
#                 else:
#                     st.success("No cards due for review! 🎉")
#                     if st.button("End Review"):
#                         st.session_state.reviewing = False
#                         st.experimental_rerun()
#     else:
#         st.info("Please upload a PDF file to begin.")

# if __name__ == "__main__":
#     main()








