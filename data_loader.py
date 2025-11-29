import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# CRITICAL CHANGE: New directory name to force a complete rebuild.
# Streamlit Cloud will recognize this as a new data artifact and rebuild it cleanly.
PERSIST_DIR = "chroma_db_final" 

def build_vector_store():
    """Builds and persists the Chroma vector store from documents."""
    
    # 1. Load documents (looks for .docx files in HR_Policy_Docs)
    loader = DirectoryLoader(
        'HR_Policy_Docs',
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        recursive=True
    )
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("ACTION REQUIRED: Ensure you have a folder named 'HR_Policy_Docs' with at least one .docx file.")
        return None
        
    if not documents:
        print("No documents found to load. Skipping vector store creation.")
        return None

    # 2. Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # 3. Create HuggingFace Embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use CPU for local environments
    )

    # 4. Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vector_store.persist()
    print(f"Vector store built and persisted to {PERSIST_DIR}")
    return vector_store

def get_vector_store():
    """Loads an existing vector store or builds a new one if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Check if the persisted store exists
    if os.path.exists(PERSIST_DIR) and os.path.isdir(PERSIST_DIR):
        try:
            vector_store = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            # Check if the collection is empty
            if vector_store._collection.count() > 0:
                # If collection has data, load it
                return vector_store
            else:
                # If store is empty, rebuild it
                return build_vector_store()
        except Exception:
            # If loading fails (e.g., corruption), rebuild
            return build_vector_store()
    else:
        # If store not found, build new one
        return build_vector_store()