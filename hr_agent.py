import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# Assumes 'data_loader.py' is in your repository and defines 'get_vector_store'
from data_loader import get_vector_store 

class HRAgent:
    """
    Retrieval-Augmented Generation (RAG) HR Policy Assistant using Gemini.

    Initializes the LLM, vector store, memory, and the RAG chain.
    """

    def __init__(self):
        # 1. Load vector store (ChromaDB)
        self.vector_store = get_vector_store()

        if not self.vector_store:
            raise ValueError(
                "Vector store failed to load. Please verify:\n"
                "1. Your Streamlit secrets file has GEMINI_API_KEY.\n"
                "2. The HR_Policy_Docs folder contains readable policy documents."
            )

        # Retrieve documents from the vector store
        self.retriever = self.vector_store.as_retriever(
            # Optional: Sets the number of documents to retrieve
            search_kwargs={"k": 3} 
        )

        # 2. API Key Check
        if not os.environ.get("GEMINI_API_KEY"):
            # This is critical for deployment on Streamlit Cloud
            raise ValueError("GEMINI_API_KEY not set in environment (Streamlit Secrets).")

        # 3. Chat Model Initialization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        # 4. Memory for chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 5. Retrieval-augmented chain setup
        # The key change: set return_source_documents=True to easily access context
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True  # MANDATORY for getting sources
        )

    def get_response(self, query: str):
        """
        Processes a user query and returns the AI response and source documents.
        """
        try:
            # Invoke the retrieval chain
            result = self.chain.invoke({"question": query})
            answer = result.get("answer", "I couldn't find an answer in the policy documents.")
            
            sources = []
            
            # Extract metadata from the retrieved documents
            # The documents are now correctly stored under the 'source_documents' key
            for doc in result.get("source_documents", []):
                meta = doc.metadata.get("source")
                if meta:
                    sources.append(meta)

            return {
                "answer": answer,
                # Use set to ensure sources are unique and list to return
                "sources": list(set(sources)) 
            }

        except Exception as e:
            # Graceful error handling
            print(f"Error during chain invocation: {e}")
            return {
                "answer": f"⚠️ An internal error occurred while processing your request: {e}",
                "sources": []
            }


if __name__ == "__main__":
    # Example test block (only runs if executed directly)
    try:
        # NOTE: For local testing, ensure 'data_loader.py' and policy files are ready, 
        # and GEMINI_API_KEY is set in your local environment.
        agent = HRAgent()
        print("[SUCCESS] HR Agent Ready for testing.")
        test_query = "How many sick days are employees entitled to?"
        response = agent.get_response(test_query)
        print(f"\nQuery: {test_query}")
        print(f"Answer: {response['answer']}")
        print(f"Sources: {', '.join(response['sources'])}")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to initialize agent: {e}")