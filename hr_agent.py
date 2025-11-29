import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_loader import get_vector_store

class HRAgent:
    """Retrieval-Augmented HR Policy Assistant."""

    def __init__(self):
        # Load vector store
        self.vector_store = get_vector_store()

        if not self.vector_store:
            raise ValueError(
                "Vector store failed to load. Check:\n"
                "- Streamlit Secrets: GEMINI_API_KEY\n"
                "- Folder HR_Policy_Docs contains at least one file"
            )

        self.retriever = self.vector_store.as_retriever()

        # Check API key
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not set in environment.")

        # Chat model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        # Memory for chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Retrieval-augmented chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def get_response(self, query: str):
        """Handles queries + returns sources."""
        try:
            result = self.chain.invoke({"question": query})
            answer = result.get("answer", "No response available.")
            sources = []

            # Extract metadata for citations (if available)
            for ctx in result.get("context", []):
                meta = ctx.metadata.get("source")
                if meta:
                    sources.append(meta)

            return {
                "answer": answer,
                "sources": list(set(sources))
            }

        except Exception as e:
            return {
                "answer": f"⚠️ Error processing request: {e}",
                "sources": []
            }


if __name__ == "__main__":
    try:
        agent = HRAgent()
        print("[SUCCESS] HR Agent Ready.")
        print(agent.get_response("What is the leave policy?"))
    except Exception as e:
        print("[ERROR]", e)
