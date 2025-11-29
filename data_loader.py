import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# IMPORTANT: Updated to match your real folder name
DATA_PATH = "HR_Policy_Docs"

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """Loads PDF, DOCX, TXT files from HR_Policy_Docs."""
    documents = []

    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"[WARNING] Folder '{data_path}' is missing or empty.")
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
                print(f"[OK] Loaded: {file_name}")
            except Exception as e:
                print(f"[ERROR] Cannot load {file_name}: {e}")

    if not documents:
        print("[WARNING] No valid documents found in HR_Policy_Docs.")
        documents.append(Document(
            page_content="No HR files found. Please upload your company policies.",
            metadata={"source": "system_default"}
        ))

    return documents


def get_vector_store():
    """Creates or loads a Chroma vector database."""
    documents = load_documents()

    if not documents:
        print("[ERROR] No documents found. Vector store not created.")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY not set. Cannot create embeddings.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # Use existing DB if available
    if os.path.exists(persist_directory):
        print("[INFO] Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    # Create new DB
    print("[INFO] Creating new Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print("[INFO] Vector store created.")

    return vector_store


if __name__ == "__main__":
    vs = get_vector_store()
    if vs:
        print("[SUCCESS] Vector store is ready.")
