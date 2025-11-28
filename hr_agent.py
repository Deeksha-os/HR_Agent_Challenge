import os
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
# Imports synchronized with the stable 'langchain' package
from langchain.schema import Document 
from data_loader import get_vector_store # Imports the function that builds the in-memory DB

class HRAgent:
    def __init__(self):
        """Initialize the HR Agent with an in-memory Chroma vector store and Gemini model"""
        print("Initializing HRAgent...")
        
        # --- 1. Get Vector Store (Diskless) ---
        self.vector_store = get_vector_store() 
        
        if self.vector_store is None:
            # 🚨 FINAL EXPLICIT ERROR MESSAGE for runtime failure
            raise Exception("Failed to load vector store. Final Check: 1. Documents in 'HR_Policy_Docs' 2. GEMINI_API_KEY secret set in Streamlit.")
        
        # --- 2. Initialize Gemini LLM ---
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3
        )
        
        # --- 3. Create Retriever ---
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
            # Fallback error for any issue during retrieval or generation
            return {
                "answer": f"A critical error occurred during processing: {str(e)}. This is likely a configuration or API issue.",
                "sources": []
            }

if __name__ == '__main__':
    print("Attempting to initialize HRAgent...")
    try:
        agent = HRAgent()
        print("HRAgent initialized successfully.")
        
        test_query = "What is the policy for using sick leave?"
        print(f"\nTesting Query: {test_query}")
        result = agent.get_response(test_query)
        print(f"Test Answer: {result['answer']}")
        print(f"Test Sources: {result['sources']}")
        
    except Exception as init_error:
        print(f"Error during HRAgent initialization: {init_error}")
        print("Please ensure your 'HR_Policy_Docs' directory exists and contains documents.")