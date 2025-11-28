import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# CRITICAL CHANGE 1: Import from the new, dedicated package
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# CRITICAL CHANGE 2: Import FAISS and remove old Chroma import
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# CRITICAL CHANGE 3: Import Document from the new core
from langchain_core.documents import Document 

# --- Configuration ---
DATA_DIR = "HR_Policy_Docs" # Assuming your policy files are in the 'policies' subfolder
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
    Creates an in-memory FAISS vector store from the policy documents.
    """
    print("Starting FAISS vector store creation in memory...")
    
    # 1. Load Documents
    raw_documents = load_all_documents()
    if not raw_documents:
        print("WARNING: No documents were loaded. Vector store will be empty.")
        return None

    # 2. Split Documents
    # Uses the chunking configuration defined above
    text_splitter = RecursiveCharacterTextSplitter(**CHUNKING_CONFIG)
    chunked_documents = text_splitter.split_documents(raw_documents)
    print(f"Total documents loaded: {len(raw_documents)}. Total chunks created: {len(chunked_documents)}")

    # 3. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Create In-Memory Vector Store using FAISS
    # This creates the vector store entirely in the server's memory, 
    # avoiding disk access issues (File I/O) and compilation errors (FAISS is stable).
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