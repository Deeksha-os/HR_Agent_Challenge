import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain/RAG Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
# We will use TextLoader, but manually iterate over files instead of using DirectoryLoader for robustness
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (for local testing via .env)
load_dotenv() 

# CRITICAL CONSTANT: Renamed persistence directory to force a clean data rebuild
PERSIST_DIR = "chroma_db_final_txt_loader_v8" 
POLICY_DIR = 'HR_Policy_Docs'

# --- Data Loading and Vector Store Logic (Integrated) ---

def build_vector_store():
    """
    Builds and persists the Chroma vector store from documents in HR_Policy_Docs
    by manually iterating and loading each file for maximum robustness.
    """
    st.info("No existing vector store found or it was invalid. Building new vector store...")
    
    if not os.path.isdir(POLICY_DIR):
        st.error(f"Configuration Error: Directory not found: {POLICY_DIR}")
        return None
    
    # 1. Manually identify and load documents one by one
    policy_files = [
        os.path.join(POLICY_DIR, f) 
        for f in os.listdir(POLICY_DIR) 
        if f.endswith('.txt')
    ]
    
    if not policy_files:
        st.warning(f"No .txt documents found in {POLICY_DIR}. Cannot build vector store.")
        return None
        
    documents = []
    
    # Load each file explicitly
    for file_path in policy_files:
        try:
            # Use TextLoader directly on the file path with explicit encoding
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            # Report the error for the specific file that failed
            st.error(f"CRITICAL DOCUMENT ERROR: Failed to load {file_path}. Details: {e}")
            # Continue trying to load other files, but log the error
            
    if not documents:
        st.error("No documents were successfully loaded. Stopping vector store build.")
        return None
    
    st.info(f"Successfully loaded {len(documents)} documents.")

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        st.warning("Documents loaded but splitting resulted in zero chunks. Documents may be too small or empty.")
        return None


    # 3. Embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )

    # 4. Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vector_store.persist()
    st.success(f"Successfully built vector store with {len(texts)} chunks.")
    return vector_store

def get_vector_store():
    """Loads an existing vector store or builds a new one if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(PERSIST_DIR) and os.path.isdir(PERSIST_DIR):
        try:
            vector_store = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            # Check if the collection is not empty
            if vector_store._collection.count() > 0:
                st.success("Vector store loaded successfully from disk.")
                return vector_store
            else:
                st.warning("Existing vector store was empty. Rebuilding...")
                return build_vector_store()
        except Exception:
            # If loading fails (e.g., corruption), rebuild
            st.error("Error loading existing vector store. Rebuilding...")
            return build_vector_store()
    else:
        return build_vector_store()

# --- HR Agent Logic (Integrated within app.py) ---

class HRAgent:
    def __init__(self):
        # 0. API Key Retrieval: 
        gemini_api_key_value = None
        try:
            # Prefer Streamlit secrets or OS environment
            gemini_api_key_value = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = gemini_api_key_value 
        except Exception:
             gemini_api_key_value = os.getenv("GOOGLE_API_KEY")

        if not gemini_api_key_value:
            raise ValueError("API Key not found. Please set GEMINI_API_KEY in Streamlit secrets or GOOGLE_API_KEY in your local .env file.")

        # 1. Load vector store
        self.vector_store = get_vector_store()

        if not self.vector_store or self.vector_store._collection.count() == 0:
            raise ValueError(
                "Vector store is empty or failed to load. The app cannot function without policy documents. Check HR_Policy_Docs and confirm files have content."
            )

        # Set search parameters
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) 

        # 2. Chat Model Initialization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=gemini_api_key_value
        )

        # 3. Memory for chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 4. Retrieval-augmented chain setup
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key='answer' # Explicitly setting output key
        )

    def get_response(self, query: str):
        """Processes a user query and returns the AI response and source documents."""
        
        # --- High-Risk Keyword Filtering ---
        high_risk_keywords = ["harassment", "discrimination", "lawsuit", "legal action", "termination", "formal complaint"]
        if any(keyword in query.lower() for keyword in high_risk_keywords):
            return {
                "answer": (
                    "🚨 **Policy Violation / High-Risk Inquiry Detected** 🚨\n\n"
                    "Your question involves a sensitive topic that requires immediate, confidential, and human intervention. "
                    "Please contact the HR Director directly at [HR_EMAIL@yourcompany.com] or call the confidential hotline at [XXX-XXX-XXXX]."
                ),
                "sources": []
            }
        
        try:
            result = self.chain.invoke({"question": query})
            answer = result.get("answer", "I couldn't find an answer in the policy documents.")
            
            sources = []
            for doc in result.get("source_documents", []):
                meta = doc.metadata.get("source")
                if meta:
                    # Extracts just the filename (e.g., 'Sick_Leave_Policy.txt')
                    filename = os.path.basename(meta)
                    sources.append(filename)

            return {
                "answer": answer,
                "sources": list(set(sources)) 
            }

        except Exception as e:
            st.error(f"Error during chain invocation: {e}")
            return {
                "answer": f"⚠️ An internal error occurred while processing your request. Please check the console logs.",
                "sources": []
            }

# --- Streamlit Main App Execution ---

@st.cache_resource(show_spinner="Initializing HR Agent and loading Vector Store...")
def load_hr_agent():
    """Loads and initializes the HR Agent once."""
    try:
        return HRAgent()
    except ValueError as e:
        # This catches errors from the __init__ if vector store or API key fail
        st.error(f"Initialization Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        st.stop()


# --- Streamlit UI Setup ---
st.set_page_config(page_title="HR Policy Assistant", layout="centered")
st.title("🤖 HR Policy Assistant")
st.caption("Policy information is powered by the Gemini 2.5 Flash RAG chain.")
st.subheader("Ask me anything about HR policies, leave rules, benefits, or workplace guidance.")

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
            
            response_data = agent.get_response(prompt)
            answer = response_data["answer"]
            sources = response_data["sources"]
            
            full_response = answer
            if sources:
                source_list = "\n".join([f"- `{source}`" for source in sources])
                full_response += f"\n\n---\n\n**Sources Used (from HR_Policy_Docs):**\n{source_list}"
            
            st.markdown(full_response, unsafe_allow_html=True)
    
    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})