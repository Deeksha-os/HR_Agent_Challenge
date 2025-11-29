import streamlit as st
# Import the fixed agent file
from hr_assistant import HRAgent

# --- Initialization and Caching ---
# Use st.cache_resource to ensure the expensive RAG setup runs only once
@st.cache_resource(show_spinner=True)
def load_hr_agent():
    """Loads and initializes the HR Agent once."""
    try:
        # The HRAgent initialization includes loading the LLM and the vector store
        return HRAgent()
    except ValueError as e:
        # This catches errors like missing API key or failed vector store load
        st.error(f"Initialization Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        st.stop()


# --- Streamlit UI Setup ---
st.set_page_config(page_title="HR Policy Assistant", layout="centered")

st.title("🤖 HR Policy Assistant (Final Deployment)")

st.subheader("Ask me anything about HR policies, leave rules, benefits, or workplace guidance.")

# Load the agent (will only run once)
agent = load_hr_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Enter your question about HR policies..."):
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating response..."):
            
            # Get the response from the HR agent
            response_data = agent.get_response(prompt)
            
            answer = response_data["answer"]
            sources = response_data["sources"]
            
            # Format the answer with sources for clarity
            full_response = answer
            if sources:
                source_list = "\n".join([f"- `{source}`" for source in sources])
                full_response += f"\n\n---\n\n**Sources Used:**\n{source_list}"
            
            st.markdown(full_response, unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})