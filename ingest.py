import os
import sys
from dotenv import load_dotenv
# We import the core functions from the data_loader file
from data_loader import build_vector_store

if __name__ == "__main__":
    # Load environment variables (like GOOGLE_API_KEY if needed later)
    load_dotenv() 
    
    print("--- HR Policy Assistant Ingestion ---")
    print("Starting document ingestion and vector store construction...")
    
    # Call the function defined in data_loader.py to load, chunk, and embed documents
    vector_store = build_vector_store()
    
    if vector_store:
        collection_count = vector_store._collection.count()
        print(f"\n✅ Ingestion complete! Vector store successfully built.")
        print(f"   -> Persisted to directory: {os.path.abspath('chroma_db_final')}")
        print(f"   -> Total documents processed: {collection_count} chunks.")
        
        # You can now safely run: streamlit run app.py
    else:
        print("\n❌ Ingestion failed. Please check the following:")
        print("   1. Ensure the folder 'HR_Policy_Docs' exists.")
        print("   2. Ensure it contains at least one valid .docx file.")
        sys.exit(1)