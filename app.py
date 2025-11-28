# app.py
import streamlit as st
from hr_agent import HRAgent
from typing import List

# Set page config
st.set_page_config(
    page_title="🤖 HR Policy Assistant",
    page_icon="💼",
    layout="centered"
)

# High-risk keywords for escalation
HIGH_RISK_KEYWORDS = [
    'formal complaint', 
    'harassment', 
    'legal action', 
    'termination', 
    'discrimination'
]

# Escalation message
ESCALATION_MESSAGE = """
**Your query contains sensitive keywords that require direct human attention.**
\nPlease contact a human HR representative immediately at HR-Support@company.com or call extension 555.
"""

def contains_high_risk_keywords(text: str) -> bool:
    """Check if the text contains any high-risk keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in HIGH_RISK_KEYWORDS)

def get_hr_agent():
    return HRAgent()

def main():
    st.write("--- DEBUG: Streamlit is Running ---")
    st.title("🤖 HR Policy Assistant")
    st.write("Ask me anything about company policies and HR-related questions.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check for high-risk keywords
        if contains_high_risk_keywords(prompt):
            response = ESCALATION_MESSAGE
            sources = []
        else:
            # Get response from HR Agent
            with st.spinner("Thinking..."):
                hr_agent = get_hr_agent()
                result = hr_agent.get_response(prompt)
                response = result["answer"]
                sources = result.get("sources", [])
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()