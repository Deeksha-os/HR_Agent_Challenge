import os
from typing import List, Union
# We use community loaders and Chroma because they are wrappers for the underlying libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# CRITICAL FIX: Using Google's embeddings for speed
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# --- Configuration ---
DATA_DIR = "policies" # Assumes your policy files are in a 'policies' subfolder
CHUNKING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200
}
EMBEDDING_MODEL = "embedding-001"

def load_all_documents(directory: str = DATA_DIR) -> List[Document]:
    """Loads documents from a directory supporting PDF, DOCX, and TXT."""
    documents = []
    print(f"Loading documents from directory: {directory}")
    
    # Handle the case where the policies directory might not exist
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return documents

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        loader = None
        
        # Determine the correct loader based on file extension
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
        
        if loader:
            print(f"  -> Loading {filename}")
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return documents

def get_vector_store():
    """
    Creates an in-memory Chroma vector store from the policy documents.
    This function avoids the file permission issue by not using 'persist_directory'.
    """
    print("Starting vector store creation in memory...")
    
    # 1. Load Documents
    raw_documents = load_all_documents()
    if not raw_documents:
        print("WARNING: No documents were loaded. Vector store will be empty.")
        return None

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(**CHUNKING_CONFIG)
    chunked_documents = text_splitter.split_documents(raw_documents)
    print(f"Total documents loaded: {len(raw_documents)}. Total chunks created: {len(chunked_documents)}")

    # 3. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Create In-Memory Vector Store
    # This vector store is purely held in memory and avoids all file permission issues.
    vector_store = Chroma.from_documents(
        documents=chunked_documents, 
        embedding=embeddings
    )
    
    print("In-memory vector store successfully created.")
    return vector_store

if __name__ == '__main__':
    # Test the vector store creation locally
    store = get_vector_store()
    if store:
        test_query = "What is the policy for sick days?"
        results = store.similarity_search(test_query, k=1)
        print(f"\nTest Query Results (First Chunk):\n{results[0].page_content[:200]}...")