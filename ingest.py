import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from typing import List

def load_documents(directory: str) -> List[dict]:
    documents = []

    loaders = {
        ".pdf": lambda p: PyPDFLoader(p).load(),
        ".docx": lambda p: Docx2txtLoader(p).load(),
        ".txt": lambda p: TextLoader(p).load(),
    }

    for file_path in Path(directory).glob("*"):
        ext = file_path.suffix.lower()
        if ext in loaders:
            try:
                print(f"Loading {file_path}...")
                docs = loaders[ext](str(file_path))
                documents.extend(docs)
                print(f"Loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents


def main():
    directory = "HR_Policy_Docs"
    persist_dir = "chroma_db"

    os.makedirs(persist_dir, exist_ok=True)

    print("Loading documents...")
    documents = load_documents(directory)

    if not documents:
        print("❌ No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building vector store...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    db.persist()
    print(f"✅ Vector DB stored in {persist_dir}")


if __name__ == "__main__":
    main()
