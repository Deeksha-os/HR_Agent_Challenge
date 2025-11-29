import os
from dotenv import load_dotenv
import streamlit as st 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_loader import get_vector_store 

# Load environment variables from .env file for Streamlit Cloud deployment
load_dotenv() 

class HRAgent:
    """
    Retrieval-Augmented Generation (RAG) HR Policy Assistant using Gemini Chat.
    """

    def __init__(self):
        # 0. Get the API Key from Streamlit Secrets or environment variable
        try:
            # Try Streamlit secrets first (for deployment)
            gemini_api_key_value = st.secrets["GEMINI_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = gemini_api_key_value
        except KeyError:
             # Fallback to local .env file (for local testing)
             gemini_api_key_value = os.getenv("GOOGLE_API_KEY")

        if not gemini_api_key_value:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not found. Check Streamlit secrets or .env file.")

        # 1. Load vector store (using the new name from data_loader.py)
        self.vector_store = get_vector_store()

        if not self.vector_store:
            raise ValueError(
                "Vector store failed to load. Please ensure HR_Policy_Docs folder is correct."
            )

        # Retriever setup
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3} 
        )

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
        # *** THIS IS THE CRITICAL FIX: output_key='answer' ***
        # This tells the chain exactly which output key to use, fixing the error.
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key='answer' 
        )
        # -------------------------------------------------------------------

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
            # Use chain.invoke()
            result = self.chain.invoke({"question": query})
            answer = result.get("answer", "I couldn't find an answer in the policy documents.")
            
            sources = []
            
            # Extract and process source metadata
            for doc in result.get("source_documents", []):
                meta = doc.metadata.get("source")
                if meta:
                    # Clean up the path to only show the filename
                    filename = os.path.basename(meta)
                    sources.append(filename)

            return {
                "answer": answer,
                "sources": list(set(sources)) # Remove duplicates
            }

        except Exception as e:
            print(f"Error during chain invocation: {e}")
            return {
                "answer": f"⚠️ An internal error occurred while processing your request: {e}",
                "sources": []
            }