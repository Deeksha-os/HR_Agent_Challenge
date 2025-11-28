# app.py
import streamlit as st
from hr_agent import HRAgent
import re

# Set page config
st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="centered"
)

# Initialize session state
if 'hr_agent' not in st.session_state:
    st.session_state.hr_agent = HRAgent()

# High-risk keywords for escalation
HIGH_RISK_KEYWORDS = [
    'formal complaint', 
    'harassment', 
    'legal action', 
    'termination', 
    'discrimination'
]

# Define escalation message
ESCALATION_MESSAGE = """
**Your query contains sensitive keywords that require direct human attention.**
\nPlease contact a human HR representative immediately at HR-Support@company.com or call extension 555.
"""

def contains_high_risk_keywords(text: str) -> bool:
    """Check if the text contains any high-risk keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in HIGH_RISK_KEYWORDS)

# UI Components
st.title("🤖 HR Policy Assistant")
st.write("Ask me anything about company policies and HR-related questions.")

# Chat input
user_query = st.text_area(
    "Your question:",
    placeholder="Type your HR-related question here...",
    height=100
)

# Submit button
if st.button("Submit", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question..."):
            # Check for high-risk keywords
            if contains_high_risk_keywords(user_query):
                st.warning(ESCALATION_MESSAGE)
            else:
                try:
                    # Get response from HR Agent
                    response = st.session_state.hr_agent.get_response(user_query)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(response["answer"])
                    
                    # Display sources if available
                    if response.get("sources"):
                        st.subheader("Sources:")
                        for source in response["sources"]:
                            st.write(f"- {source}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Add some styling
st.markdown("""
<style>
    .stTextArea [data-baseweb=base-input] {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)