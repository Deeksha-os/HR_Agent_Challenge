import os
from typing import List
# FIX: Document loaders and the vector store were moved to 'langchain_community'
# in the version 0.2.x of the LangChain library.
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # <-- FIX: Moved from langchain.vectorstores
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document # Using core package for stability

# Define the directory where your HR documents are stored
DATA_PATH = "hr_documents" 

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Loads all supported documents (PDF, DOCX, TXT) from the specified path.
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
    
    # Fallback if no documents were found
    if not documents:
        print("No supported documents found or loaded.")
        # Create a mock document to prevent downstream errors if the app relies on the vector store being created
        documents.append(Document(page_content="Welcome to the HR Assistant. Upload your policies to begin.", metadata={"source": "system_default"}))

    return documents

def get_vector_store():
    """
    Loads documents, splits them, creates embeddings, and initializes the Chroma vector store.
    """
    documents = load_documents()
    
    if not documents:
        # If no documents are loaded, we can't create a vector store.
        return None

    # 1. Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return None

    # 2. Create the embeddings model
    # The default model is 'gemini-2.5-flash' for embeddings with this library version.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Create the Chroma vector store
    # We use a persistent directory to save the vector store across runs
    persist_directory = "chroma_db"
    
    # Check if a persistent database exists
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
        vector_store.persist() # Save the new store
        print("New vector store created and persisted.")
        
    return vector_store

if __name__ == "__main__":
    # This part is just for testing the data loading process locally
    # Note: Requires a real 'hr_documents' folder and GEMINI_API_KEY
    vector_store = get_vector_store()
    if vector_store:
        print(f"Vector store successfully initialized with {vector_store._collection.count()} documents.")