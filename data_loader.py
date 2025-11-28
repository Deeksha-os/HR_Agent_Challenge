import os
from typing import List
# We use langchain_community for document loaders and vector stores 
# to be compatible with our pinned version set in requirements.txt
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document # Standardized import for Document object

# Define the directory where your HR documents are stored
DATA_PATH = "hr_documents" 

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Loads all supported documents (PDF, DOCX, TXT) from the specified path.
    Handles warnings if the data directory is missing or empty.
    """
    documents = []
    
    # Check if the directory exists and is not empty
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"Warning: Data directory '{data_path}' not found or is empty. Cannot load HR documents.")
        return []

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        loader = None

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        
        if loader:
            try:
                documents.extend(loader.load())
                print(f"Successfully loaded: {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    # Fallback: If no actual documents are found, create a placeholder document
    if not documents:
        print("No supported documents found or loaded.")
        documents.append(Document(page_content="Welcome to the HR Assistant. Please upload your policy documents to begin the RAG process.", metadata={"source": "system_default"}))

    return documents

def get_vector_store():
    """
    Loads documents, splits them, creates embeddings, and initializes the Chroma vector store.
    It uses a persistent directory to save the vector store across runs.
    """
    documents = load_documents()
    
    if not documents:
        # Cannot proceed if document loading failed entirely
        return None

    # 1. Split the documents into smaller chunks for effective retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set. Cannot create embeddings.")
        return None

    # 2. Create the embeddings model using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Create/Load the Chroma vector store
    persist_directory = "chroma_db"
    
    # Check if a persistent database exists to avoid re-indexing policies every time
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist() # Save the new store to disk
        print("New vector store created and persisted.")
        
    return vector_store

if __name__ == "__main__":
    # Example usage for local testing
    vector_store = get_vector_store()
    if vector_store:
        print(f"Vector store successfully initialized with {vector_store._collection.count()} documents.")