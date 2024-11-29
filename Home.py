import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Reference dataset for techniques
reference_data = {
    'Student ID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 
                  'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'],
    'Performance Level': ['Low Performance', 'Medium Performance', 'High Performance', 
                         'Low Performance', 'Medium Performance', 'Low Performance',
                         'High Performance', 'Low Performance', 'Medium Performance',
                         'Low Performance', 'High Performance', 'Medium Performance',
                         'Low Performance', 'Medium Performance', 'High Performance',
                         'Medium Performance', 'High Performance', 'Low Performance',
                         'Medium Performance', 'High Performance'],
    'Recommended Technique': ['Leitner', 'Pomodoro', 'Feynman', 'Leitner', 'Pomodoro',
                            'Leitner', 'Feynman', 'Leitner', 'Pomodoro', 'Leitner',
                            'Feynman', 'Pomodoro', 'Leitner', 'Pomodoro', 'Feynman',
                            'Pomodoro', 'Feynman', 'Leitner', 'Pomodoro', 'Feynman']
}

technique_descriptions = {
    'Leitner': """
    **Leitner System** - Flashcard-based learning system:
    - Break down information into small, manageable pieces
    - Use flashcards with increasing review intervals
    - Move cards between boxes based on correct/incorrect answers
    - Helps with memory retention and focus
    
    Best for: People who need structure and frequent reinforcement of concepts.
    """,
    
    'Pomodoro': """
    **Pomodoro Technique** - Time management method:
    - Work for 25 minutes focused sessions
    - Take 5-minute breaks between sessions
    - Take longer breaks (15-30 minutes) after 4 sessions
    - Track your progress and adjust timing as needed
    
    Best for: People who struggle with time management and need regular breaks.
    """,
    
    'Feynman': """
    **Feynman Technique** - Concept mastery through teaching:
    - Choose a concept to learn
    - Explain it in simple terms as if teaching someone else
    - Identify gaps in your explanation
    - Review and simplify explanation
    
    Best for: People who learn best through active engagement and explanation.
    """
}

def get_recommended_technique(performance_level):
    if performance_level == "Low Performance (High ADHD tendencies)":
        return "Leitner"
    elif performance_level == "Medium Performance":
        return "Pomodoro"
    else:  # High Performance
        return "Feynman"

def get_performance_level(total_score):
    if 30 <= total_score <= 40:
        return "Low Performance (High ADHD tendencies)", """
        Your score indicates higher ADHD tendencies. Common characteristics might include:
        - Difficulty maintaining focus on tasks
        - Challenges with organization and time management
        - Tendency to procrastinate
        Please consult with a healthcare professional for a proper evaluation.
        """
    elif 21 <= total_score <= 29:
        return "Medium Performance", """
        Your score indicates moderate ADHD tendencies. You may experience:
        - Occasional difficulties with focus
        - Some challenges with organization
        - Moderate procrastination tendencies
        Consider discussing these symptoms with a healthcare provider if they impact your daily life.
        """
    else:  # 10-20
        return "High Performance (Low ADHD tendencies)", """
        Your score indicates lower ADHD tendencies. You typically:
        - Maintain focus well
        - Have good organizational skills
        - Manage time effectively
        """

def main():
    st.title("Student Study Behavior Survey")
    st.write("Please answer the following questions honestly based on your experiences.")
    
    # Dictionary containing all questions and their options
    questions = {
        "1. How often do you struggle to stay focused on tasks that are not immediately interesting to you?": [
            "Rarely", "Sometimes", "Often", "Very often"
        ],
        "2. How frequently do you have difficulty completing assignments on time because of procrastination?": [
            "Rarely", "Sometimes", "Often", "Almost always"
        ],
        "3. When you are in a classroom setting, how likely are you to lose focus due to external distractions?": [
            "Not at all likely", "Slightly likely", "Moderately likely", "Extremely likely"
        ],
        "4. How often do you forget to bring important materials (e.g., books, assignments, stationery) to school or university?": [
            "Almost never", "Occasionally", "Frequently", "Very frequently"
        ],
        "5. When studying or doing homework, how often do you find yourself daydreaming or losing track of time?": [
            "Rarely", "Sometimes", "Often", "Very often"
        ],
        "6. How often do you feel restless or fidgety when you are expected to stay seated for a prolonged period?": [
            "Rarely", "Sometimes", "Often", "Almost always"
        ],
        "7. How well do you organize your assignments and materials in a way that you can easily find them?": [
            "Very well", "Fairly well", "Poorly", "Very poorly"
        ],
        "8. How likely are you to start multiple projects or assignments but have difficulty finishing them?": [
            "Not at all likely", "Slightly likely", "Moderately likely", "Very likely"
        ],
        "9. How often do you need to re-read material or go over instructions multiple times to understand them?": [
            "Rarely", "Sometimes", "Often", "Very often"
        ],
        "10. When you need to make quick decisions, how often do you act impulsively without considering consequences?": [
            "Almost never", "Occasionally", "Frequently", "Very frequently"
        ]
    }
    
    # Store responses
    responses = {}
    
    # Create radio buttons for each question
    for question, options in questions.items():
        responses[question] = st.radio(question, options, key=question)
    
    # Submit button
    if st.button("Submit Questionnaire"):
        # Calculate score
        total_score = 0
        detailed_scores = []
        
        for question, answer in responses.items():
            options = questions[question]
            score = options.index(answer) + 1
            total_score += score
            detailed_scores.append(score)
        
        # Get performance level and recommended technique
        performance_level, description = get_performance_level(total_score)
        recommended_technique = get_recommended_technique(performance_level)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Results Summary", "Detailed Analysis", "Study Recommendations"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Score", total_score)
            with col2:
                st.metric("Performance Level", performance_level)
            with col3:
                st.metric("Recommended Technique", recommended_technique)
            
            st.write(description)
        
        with tab2:
            # Create a bar chart of responses
            st.subheader("Response Profile")
            import matplotlib.pyplot as plt
            
            # Assuming `detailed_scores` is already defined
            fig, ax = plt.subplots(figsize=(20, 6))
            questions_short = [f"Q{i+1}" for i in range(10)]
            ax.plot(questions_short, detailed_scores, marker='o', linestyle='-', color='g')
            ax.set_ylabel("Score (1-4)")
            ax.set_xlabel("Questions")
            ax.set_title("Score Distribution Across Questions")
            ax.set_ylim(0, 5)
            st.pyplot(fig)

            
            # Display response summary
            st.subheader("Your Responses:")
            df = pd.DataFrame({
                "Question": list(responses.keys()),
                "Your Answer": list(responses.values()),
                "Score (1-4)": detailed_scores
            })
            st.dataframe(df)
        
        with tab3:
            st.header(f"Recommended Study Technique: {recommended_technique}")
            st.markdown(technique_descriptions[recommended_technique])
            
            # Display success statistics
            ref_df = pd.DataFrame(reference_data)
            success_count = len(ref_df[ref_df['Recommended Technique'] == recommended_technique])
            st.info(f"This technique has been recommended to {success_count} out of 20 students in our reference group with similar performance levels.")
            
            # Show distribution of techniques
            st.subheader("Technique Distribution by Performance Level")
            technique_dist = ref_df.groupby(['Performance Level', 'Recommended Technique']).size().unstack()
            st.bar_chart(technique_dist)
        
        st.write("""
        **Important Note:** This questionnaire is for screening purposes only and does not constitute a medical diagnosis. 
        If you have concerns about ADHD, please consult with a qualified healthcare professional for a proper evaluation.
        """)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="adhd_screening_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()