import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document # Use langchain_core for Document import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_loader import get_vector_store 

class HRAgent:
    """
    An HR Agent that answers questions based on a vector store of HR documents.
    """
    def __init__(self):
        # 1. Initialize the vector store and retriever
        self.vector_store = get_vector_store()
        
        if not self.vector_store:
            # Propagate the error if the vector store failed to load/create
            raise ValueError("Vector store failed to initialize. Check data_loader.py logs.")
            
        self.retriever = self.vector_store.as_retriever()
        
        # 2. Initialize the Chat Model (Gemini)
        if not os.environ.get("GEMINI_API_KEY"):
             raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Using the specified model for chat
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3
        )

        # 3. Initialize Conversation Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

        # 4. Create the Conversational Retrieval Chain
        # This chain manages the RAG flow: retrieve documents, then generate response
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            # Set system instruction to guide the HR agent's behavior
            combine_docs_chain_kwargs={
                "prompt": """
                You are a highly professional and friendly HR Assistant. 
                Use the provided context documents to answer the user's questions about HR policies.
                
                If the answer is not contained in the context, clearly state that 
                you cannot provide an answer based on the available HR documents. 
                Do not make up information.
                
                Always cite your sources by mentioning the document's source (e.g., 'Source: [metadata.source]').
                
                Context: {context}
                
                Question: {question}
                
                Answer:
                """
            }
        )

    def ask(self, query: str) -> str:
        """
        Processes a user query and returns the agent's response.
        """
        try:
            # Invoke the chain to get the answer
            result = self.chain.invoke({"question": query})
            return result.get("answer", "I couldn't process that question. Please try again.")
        except Exception as e:
            return f"An error occurred while processing your request: {e}"

# Example usage (for testing or if the app needs to directly instantiate)
if __name__ == "__main__":
    # Note: This block assumes a valid GEMINI_API_KEY is set in the environment
    # and a 'hr_documents' folder exists.
    try:
        agent = HRAgent()
        print("HR Agent initialized. Ready to chat.")
        
        response = agent.ask("What is the policy on sick leave?")
        print(f"Agent Response: {response}")
        
    except ValueError as e:
        print(f"Initialization Failed: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}")