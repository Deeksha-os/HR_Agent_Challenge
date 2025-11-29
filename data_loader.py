import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # CORRECT: Used to load HF DB
from langchain_core.documents import Document

DATA_PATH = "HR_Policy_Docs"
# Must match the model used in your ingest.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """Loads PDF, DOCX, TXT files from HR_Policy_Docs."""
    documents = []

    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"[WARNING] Folder '{data_path}' is missing or empty.")
        return []

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        loader = None

        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        
        if loader:
            try:
                documents.extend(loader.load())
                print(f"[OK] Loaded: {file_name}")
            except Exception as e:
                print(f"[ERROR] Cannot load {file_name}: {e}")

    if not documents:
        print("[WARNING] No valid documents found in HR_Policy_Docs. Returning placeholder.")
        documents.append(Document(
            page_content="No HR files found. Please upload your company policies.",
            metadata={"source": "system_default"}
        ))

    return documents


def get_vector_store():
    """Creates or loads a Chroma vector database using HuggingFace Embeddings."""
    
    # Initialize HuggingFace Embeddings (Must match ingest.py)
    print(f"[INFO] Initializing HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"} # Force CPU for cloud stability
    )

    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # Use existing DB if available (Expected path for deployment)
    if os.path.exists(persist_directory):
        print("[INFO] Loading existing Chroma vector store...")
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings # CRITICAL: Loads with correct HF embedding function
            )
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load Chroma DB: {e}. Check if embeddings match.")
            return None
    else:
        # Fallback: Create new DB if no persisted directory is found
        print("[WARNING] Chroma DB directory not found. Attempting to create a new one...")
        documents = load_documents()
        if not documents:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        print("[INFO] New vector store created.")
        return vector_store