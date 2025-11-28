import os
import warnings
from typing import Dict, List

# LangChain Imports
from langchain_community.vectorstores import Chroma
# CORRECTED: Import Google Generative AI components for hosted services
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 

# Suppress the specific LangChain deprecation warning
warnings.filterwarnings(
    "ignore", 
    message="The class `langchain.embeddings.huggingface.HuggingFaceEmbeddings` is deprecated."
)

class HRAgent:
    def __init__(self, db_path: str = "chroma_db"):
        """Initialize the HR Agent with ChromaDB and Gemini model"""
        print("Initializing HRAgent...")
        
        # --- 1. Initialize Embeddings (SPEED FIX) ---
        # Uses the hosted Google service (embedding-001) which is fast and requires no local download.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="embedding-001" 
        )
        
        # --- 2. Initialize ChromaDB Vector Store (FILE PERMISSION FIX) ---
        # Added collection_metadata={"allow_reset": True} to prevent the "unable to open database file" 
        # error caused by Streamlit Cloud's read-only file system.
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            # CRITICAL FIX: Allows Chroma to load the existing DB on the read-only server.
            collection_metadata={"allow_reset": True} 
        )
        
        # --- 3. Initialize Gemini LLM ---
        # The API key is automatically read by the LangChain integration from Streamlit's secrets.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3
        )
        
        # --- 4. Create Retriever ---
        # The retriever fetches the most relevant document chunks based on the query.
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        print("HRAgent initialized successfully.")
    
    def get_response(self, query: str) -> Dict[str, any]:
        """
        Get response for the given query using the RAG pipeline.
        """
        try:
            # --- 1. Retrieval Step ---
            docs = self.retriever.invoke(query)
            
            # --- 2. Context Formatting ---
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Extract unique source filenames for citation
            sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs]))
            
            # --- 3. Prompt Construction ---
            prompt = f"""You are a highly detailed, professional, and helpful HR assistant. Your goal is to provide comprehensive and friendly answers to employee questions. 
            Use ONLY the following context to answer the question at the end. Structure your response clearly with bullet points or paragraphs.
            If the context does not contain the answer, politely state that you could not find the specific information in the current policy documents, and suggest they contact HR for clarification.

            Context:
            {context}

            Question: {query}

            Answer:"""
            
            # --- 4. Generation Step ---
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": sources
            }
            
        except Exception as e:
            # Provide a clear, non-technical message to the user if the RAG process fails
            return {
                "answer": "I apologize, but an unexpected error occurred while processing your request. Please ensure the required policy documents are accessible and your API key is valid, then try again.",
                "sources": []
            }

if __name__ == '__main__':
    print("Attempting to initialize HRAgent...")
    try:
        agent = HRAgent()
        print("HRAgent initialized successfully.")
        
        # Test a query
        test_query = "What is the policy for using sick leave?"
        print(f"\nTesting Query: {test_query}")
        result = agent.get_response(test_query)
        print(f"Test Answer: {result['answer']}")
        print(f"Test Sources: {result['sources']}")
        
    except Exception as init_error:
        print(f"Error during HRAgent initialization: {init_error}")
        print("Please ensure your API key is correct and the chroma_db files are present.")