import os
from typing import List
# Imports synchronized with the stable 'langchain' package
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import FAISS 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document # Use langchain.schema for Document

# --- Configuration ---
# *** CRITICAL: MUST MATCH THE FOLDER NAME IN YOUR GITHUB REPO (case-sensitive) ***
DATA_DIR = "HR_Policy_Docs" 
CHUNKING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200
}
EMBEDDING_MODEL = "embedding-001"

def load_all_documents(directory: str = DATA_DIR) -> List[Document]:
    """Loads documents from a directory supporting PDF, DOCX, and TXT."""
    documents = []
    print(f"Loading documents from directory: {directory}")
    
    # 🚨 CRITICAL CHECK: Ensure the directory exists
    if not os.path.exists(directory):
        print(f"ERROR: Document directory '{directory}' not found. Check Git structure.")
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
                print(f"ERROR: Failed to load document {filename}: {e}")
                
    return documents

def get_vector_store():
    """
    Creates an in-memory FAISS vector store from the policy documents.
    """
    print("Starting FAISS vector store creation in memory...")
    
    # 1. Load Documents
    raw_documents = load_all_documents()
    if not raw_documents:
        print("CRITICAL FAILURE: No raw documents were loaded. Returning None.")
        return None

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(**CHUNKING_CONFIG)
    chunked_documents = text_splitter.split_documents(raw_documents)
    print(f"Total documents loaded: {len(raw_documents)}. Total chunks created: {len(chunked_documents)}")

    # 3. Initialize Embeddings (This is where the API key is used)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        # Force a small operation to check API key validity early
        _ = embeddings.embed_query("check") 
    except Exception as e:
        # Check 2: API key failure during embedding initialization
        print(f"CRITICAL FAILURE: Embeddings model failed to initialize. Check Streamlit Secret 'GEMINI_API_KEY'. Error: {e}")
        return None

    # 4. Create In-Memory Vector Store using FAISS
    vector_store = FAISS.from_documents(
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