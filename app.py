import streamlit as st
from hr_agent import HRAgent

# --- Force Caching Fix/LLM Initialization ---
# Use st.cache_resource for the agent instance to prevent re-initialization on every rerun
@st.cache_resource
def load_hr_agent():
    """Loads and initializes the HR Agent once."""
    try:
        return HRAgent()
    except ValueError as e:
        st.error(f"Initialization Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        st.stop()


# --- Streamlit UI Setup ---
# ***Cosmetic change to trigger full cache refresh***
st.title("🤖 HR Policy Assistant (Live Version)")

st.subheader("Ask me anything about HR policies, leave rules, benefits, or workplace guidance.")

# Load the agent (will only run once due to st.cache_resource)
agent = load_hr_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question about HR policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating response..."):
            
            # Get the response from the HR agent
            response_data = agent.get_response(prompt)
            
            answer = response_data["answer"]
            sources = response_data["sources"]
            
            # Format the answer with sources
            full_response = answer
            if sources:
                full_response += "\n\n---\n\n**Sources Used:**\n" + "\n".join(
                    [f"- `{source}`" for source in sources]
                )
            
            st.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})