import os
import streamlit as st 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_loader import get_vector_store 

class HRAgent:
    """
    Retrieval-Augmented Generation (RAG) HR Policy Assistant using Gemini Chat.
    """

    def __init__(self):
        # 0. Get the API Key for the LLM using Streamlit's stable method
        try:
            gemini_api_key_value = st.secrets["GEMINI_API_KEY"]
            
            # CRITICAL FIX: Set the value into the GOOGLE_API_KEY environment variable.
            os.environ["GOOGLE_API_KEY"] = gemini_api_key_value
            
        except KeyError:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets for LLM initialization.")

        # 1. Load vector store (ChromaDB)
        self.vector_store = get_vector_store()

        if not self.vector_store:
            raise ValueError(
                "Vector store failed to load. Please verify:\n"
                "1. The vector store was built and loaded with **HuggingFace Embeddings**.\n"
                "2. The HR_Policy_Docs folder contains readable policy documents."
            )

        # Retrieve documents from the vector store
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
        # --- FIX APPLIED HERE ---
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key='answer' # <--- THIS TELLS THE MEMORY WHICH KEY TO SAVE
        )
        # -------------------------

    def get_response(self, query: str):
        """
        Processes a user query and returns the AI response and source documents.
        """
        try:
            result = self.chain.invoke({"question": query})
            # Ensure we are extracting the key used in the chain's output
            answer = result.get("answer", "I couldn't find an answer in the policy documents.")
            
            sources = []
            
            for doc in result.get("source_documents", []):
                meta = doc.metadata.get("source")
                if meta:
                    sources.append(meta)

            return {
                "answer": answer,
                "sources": list(set(sources)) 
            }

        except Exception as e:
            print(f"Error during chain invocation: {e}")
            return {
                "answer": f"⚠️ An internal error occurred while processing your request: {e}",
                "sources": []
            }


if __name__ == "__main__":
    print("This file should be run via Streamlit (app.py).")