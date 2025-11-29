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
    "formal complaint",
    "harassment",
    "legal action",
    "termination",
    "discrimination"
]

# Escalation message
ESCALATION_MESSAGE = """
**⚠️ Sensitive Query Detected**
Your message contains keywords that require direct human attention.

Please contact a human HR representative:
📧 HR-Support@company.com  
📞 Extension: 555
"""

def contains_high_risk_keywords(text: str) -> bool:
    """Check if the text contains any high-risk keywords."""
    text_lower = text.lower()
    return any(k in text_lower for k in HIGH_RISK_KEYWORDS)

# Cached HR Agent initialization
@st.cache_resource
def get_hr_agent():
    """
    Safely initialize the HR Agent.
    Uses st.cache_resource to ensure the heavy loading (vector store)
    only happens once.
    """
    try:
        agent = HRAgent()
        return agent
    except Exception as e:
        st.error(f"❌ HR Agent failed to initialize: {e}")
        st.info("Please check:\n- Streamlit Secrets `GEMINI_API_KEY`\n- Folder `HR_Policy_Docs/` contains valid files")
        # Return None so the rest of the app knows the agent is not ready
        return None

def main():
    st.title("🤖 HR Policy Assistant")
    st.write("Ask me anything about HR policies, leave rules, benefits, or workplace guidance.")
    st.divider()

    # Initialize session chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

    # Chat input box
    if prompt := st.chat_input("Type your question…"):

        # Record user message and display it immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Step 1: High-risk keyword detection
        if contains_high_risk_keywords(prompt):
            response = ESCALATION_MESSAGE
            sources = []

        else:
            # Step 2: Load HR Agent
            with st.spinner("Thinking…"):
                hr_agent = get_hr_agent()

                if hr_agent is None:
                    response = (
                        "⚠️ HR Assistant couldn't start.\n\n"
                        "Please check:\n"
                        "- `GEMINI_API_KEY`\n"
                        "- `HR_Policy_Docs/` folder"
                    )
                    sources = []

                else:
                    # 🔥 CORRECT FUNCTION CALL: hr_agent.get_response(prompt)
                    result = hr_agent.get_response(prompt)

                    # Get data from the dictionary returned by hr_agent.py
                    response = result.get("answer", "No response available.")
                    sources = result.get("sources", [])
                    
        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
            if sources:
                with st.expander("Sources"):
                    # Use a set to display only unique source file names
                    for s in list(set(sources)): 
                        st.write(f"- {s}")

        # Save assistant message in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()