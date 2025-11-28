import os
from dotenv import load_dotenv
import warnings
from typing import Dict, List

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# We are using the Google GenAI LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI 

# Suppress LangChain deprecation warning for embeddings
warnings.filterwarnings(
    "ignore", 
    message="The class `langchain.embeddings.huggingface.HuggingFaceEmbeddings` is deprecated."
)

class HRAgent:
    def __init__(self, db_path: str = "chroma_db"):
        """Initialize the HR Agent with ChromaDB and Gemini model"""
        print("Initializing HRAgent...")
        load_dotenv()
        
        # --- 1. Initialize Embeddings ---
        self.embeddings = HuggingFaceEmbeddings(
            # Using a local, efficient Sentence Transformer model for embeddings
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # --- 2. Initialize ChromaDB Vector Store ---
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # --- 3. Initialize Gemini LLM ---
        # Reads GEMINI_API_KEY from the .env file
        # IMPORTANT: Ensure your GEMINI_API_KEY is active and funded.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3,
            api_key=os.getenv("GEMINI_API_KEY") 
        )
        
        # --- 4. Create Retriever ---
        # The retriever fetches the most relevant document chunks based on the query.
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 chunks
        )
        print("HRAgent initialized successfully.")
    
    def get_response(self, query: str) -> Dict[str, any]:
        """
        Get response for the given query using the RAG pipeline.
        
        Args:
            query (str): User's question.
            
        Returns:
            Dict: Containing the generated 'answer' and list of 'sources'.
        """
        try:
            # --- 1. Retrieval Step ---
            # This calls the VectorStoreRetriever using the stable 'invoke' method.
            docs = self.retriever.invoke(query)
            
            # --- 2. Context Formatting ---
            # Combine the retrieved document content into a single context string
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Extract unique source filenames for citation
            sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs]))
            
            # --- 3. Prompt Construction ---
            # **MODIFIED INSTRUCTION:** Encourage a detailed, conversational, and helpful tone.
            prompt = f"""You are a highly detailed, professional, and helpful HR assistant. Your goal is to provide comprehensive and friendly answers to employee questions. 
            Use ONLY the following context to answer the question at the end. Structure your response clearly with bullet points or paragraphs.
            If the context does not contain the answer, politely state that you could not find the specific information in the current policy documents, and suggest they contact HR for clarification.

            Context:
            {context}

            Question: {query}

            Answer:"""
            
            # --- 4. Generation Step ---
            # Get the final response from Gemini
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": sources
            }
            
        except Exception as e:
            # Catch API errors (like the previous billing issue) or other runtime exceptions
            return {
                "answer": f"An error occurred: {str(e)}",
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
        print("Please ensure all dependencies are installed (pip install langchain-google-genai) and your .env file is correct.")