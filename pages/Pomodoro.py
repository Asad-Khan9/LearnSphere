import streamlit as st
import time
import PyPDF2
from openai import OpenAI
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client for Llama 2
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize session state variables
if 'study_materials' not in st.session_state:
    st.session_state.study_materials = {}
if 'completed_pomodoros' not in st.session_state:
    st.session_state.completed_pomodoros = 0
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 1
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False

def count_down(ts, session_type="Study"):
    """Enhanced countdown function with progress bar and session tracking"""
    placeholder = st.empty()
    progress_bar = st.progress(0)
    
    initial_time = ts
    while ts:
        mins, secs = divmod(ts, 60)
        time_now = '{:02d}:{:02d}'.format(mins, secs)
        
        with placeholder:
            st.header(f"{session_type} Session: {time_now}")
        
        # Update progress bar
        progress = 1 - (ts / initial_time)
        progress_bar.progress(progress)
        
        time.sleep(1)
        ts -= 1
    
    # Session completion actions
    if session_type == "Study":
        st.session_state.completed_pomodoros += 1
        if st.session_state.completed_pomodoros % 4 == 0:
            st.session_state.current_stage += 1
    
    placeholder.empty()
    progress_bar.empty()
    st.success(f"{session_type} Session Completed!")
    
    # Play sound notification (if supported by browser)
    st.balloons()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_llm_response(prompt, stream=True):
    messages = [
        {"role": "system", "content": "You are an expert study assistant helping students understand complex topics."},
        {"role": "user", "content": prompt}
    ]
    
    completion = client.chat.completions.create(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        messages=messages,
        temperature=0.7,
        stream=stream
    )
    
    if stream:
        return completion
    else:
        return completion.choices[0].message.content

def create_study_summary(text):
    prompt = f"""Create a concise summary of the following study material, highlighting key concepts and main points:

Text: {text[:2000]}  # Limiting text length for API

Summary:"""
    return generate_llm_response(prompt, stream=False)

def create_quiz_questions(text):
    prompt = f"""Generate 3 quiz questions based on this study material. Include multiple choice options and the correct answer:

Text: {text[:2000]}

Format each question as:
Q[number]: [Question]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]"""
    return generate_llm_response(prompt, stream=False)

def create_study_notes(text):
    prompt = f"""Create detailed study notes from this material, organizing key concepts and including examples:

Text: {text[:2000]}

Notes:"""
    return generate_llm_response(prompt, stream=False)

def main():
    st.title("📚 Pomodoro Study Assistant")
    
    # Sidebar for study material management
    with st.sidebar:
        st.header("📑 Study Materials")
        uploaded_file = st.file_uploader("Upload Study Material (PDF)", type="pdf")
        
        if uploaded_file:
            if uploaded_file.name not in st.session_state.study_materials:
                with st.spinner("Processing study material..."):
                    text = extract_text_from_pdf(uploaded_file)
                    st.session_state.study_materials[uploaded_file.name] = {
                        'text': text,
                        'summary': create_study_summary(text),
                        'quiz': create_quiz_questions(text),
                        'notes': create_study_notes(text)
                    }
                st.success(f"Added: {uploaded_file.name}")
        
        st.divider()
        st.subheader("📊 Study Statistics")
        st.metric("Completed Pomodoros", st.session_state.completed_pomodoros)
        st.metric("Current Stage", f"Stage {st.session_state.current_stage}")
    
    # Main content area
    tab1, tab2 = st.tabs(["⏱️ Pomodoro Timer", "📖 Study Materials"])
    
    with tab1:
        st.markdown("<h3 style='text-align: center;'>Pomodoro Timer</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            study_time = st.number_input(
                "Study Session (minutes)",
                min_value=1,
                max_value=60,
                value=25
            )
            
            if st.button("Start Study Session", use_container_width=True):
                count_down(int(study_time * 60), "Study")
        
        with col2:
            break_time = st.number_input(
                "Break Time (minutes)",
                min_value=1,
                max_value=30,
                value=5
            )
            
            if st.button("Start Break", use_container_width=True):
                count_down(int(break_time * 60), "Break")
        
        # Display study tips during session
        with st.expander("Study Tips"):
            st.markdown("""
            - 🎯 Focus on one task at a time
            - 💧 Stay hydrated during breaks
            - 👀 Use the 20-20-20 rule for eye strain
            - 🧘‍♂️ Take deep breaths if feeling overwhelmed
            - ✍️ Take brief notes during study sessions
            """)
    
    with tab2:
        if st.session_state.study_materials:
            selected_material = st.selectbox(
                "Select Study Material",
                list(st.session_state.study_materials.keys())
            )
            
            material = st.session_state.study_materials[selected_material]
            
            study_tab1, study_tab2, study_tab3 = st.tabs(["📝 Summary", "❓ Quiz", "📚 Notes"])
            
            with study_tab1:
                st.markdown(material['summary'])
            
            with study_tab2:
                st.markdown(material['quiz'])
                if st.button("Generate New Questions"):
                    with st.spinner("Generating new questions..."):
                        material['quiz'] = create_quiz_questions(material['text'])
                        st.rerun()
            
            with study_tab3:
                st.markdown(material['notes'])
                if st.button("Generate New Notes"):
                    with st.spinner("Generating new notes..."):
                        material['notes'] = create_study_notes(material['text'])
                        st.rerun()
        else:
            st.info("Upload study materials to get started!")

if __name__ == "__main__":
    st.set_page_config(page_title="Pomodoro Study Assistant", page_icon="📚", layout="wide")
    main()