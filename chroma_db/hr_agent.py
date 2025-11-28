# hr_agent.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

class HRAgent:
    def __init__(self, db_path="chroma_db"):
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Initialize ChromaDB
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # Initialize Claude model
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.3,
            max_tokens=1024,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def get_response(self, query: str) -> dict:
        """
        Get response for the given query using RAG
        
        Args:
            query (str): User's question
            
        Returns:
            dict: Dictionary containing 'answer' and 'sources'
        """
        try:
            # Get response from QA chain
            result = self.qa_chain.invoke({"query": query})
            
            # Extract sources
            sources = list(set(
                doc.metadata.get('source', 'Unknown') 
                for doc in result.get('source_documents', [])
            ))
            
            return {
                "answer": result.get('result', "I couldn't find an answer to your question."),
                "sources": sources
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "sources": []
            }