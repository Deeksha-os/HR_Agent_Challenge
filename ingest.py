from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# The new, working line:
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Find this line:
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Change it to this:
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from typing import List
from pathlib import Path

def load_documents(directory: str) -> List[dict]:
    """
    Load all PDF and DOCX files from the specified directory.
    
    Args:
        directory (str): Path to the directory containing documents
        
    Returns:
        List[dict]: List of loaded documents with metadata
    """
    documents = []
    
    # Supported file extensions and their corresponding loaders
    loaders = {
    '.pdf': lambda path: PyPDFLoader(path).load(),
    '.docx': lambda path: Docx2txtLoader(path).load(),
    '.txt': lambda path: TextLoader(path).load() 
}
    
    # Iterate through all files in the directory
    for file_path in Path(directory).glob('*'):
        file_ext = file_path.suffix.lower()
        if file_ext in loaders:
            try:
                print(f"Loading {file_path}...")
                docs = loaders[file_ext](str(file_path))
                documents.extend(docs)
                print(f"Successfully loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return documents

def main():
    # Configuration
    input_directory = "HR_Policy_Docs"
    persist_directory = "chroma_db"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Create output directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    print("Starting document ingestion...")
    
    # 1. Load documents
    print("\nLoading documents...")
    documents = load_documents(input_directory)
    
    if not documents:
        print("No documents found or loaded. Exiting...")
        return
    
    print(f"\nLoaded {len(documents)} document(s)")
    
    # 2. Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # 3. Initialize embeddings
    print("\nInitializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # 4. Create and persist the vector store
    print("\nCreating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # 5. Persist the database
    vector_store.persist()
    print(f"\nVector store created and persisted to '{persist_directory}'") 
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    main()
