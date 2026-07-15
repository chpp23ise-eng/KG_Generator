# import sqlite3
# print(sqlite3.sqlite_version)

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all TXT and PDF files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your files."
        )
    
    documents = []

    # ✅ Load TXT files (same as before)
    txt_loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents.extend(txt_loader.load())

    # ✅ Load PDF files (NEW)
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents.extend(pdf_loader.load())

    # ❌ If nothing found
    if len(documents) == 0:
        raise FileNotFoundError(
            f"No TXT or PDF files found in {docs_path}. Please add documents."
        )

    # 🔍 Debug print (same as your original)
    for i, doc in enumerate(documents[:3]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    print(f"\n✅ Total documents loaded: {len(documents)}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n✅ Total chunks created: {len(chunks)}")

    # 🔍 Better debug view
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Page: {chunk.metadata.get('page', 'N/A')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Preview: {chunk.page_content[:150]}...")
        print("-" * 50)
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    
    print("--- Creating vector store ---")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # ✅ Ensure it's saved
    # vectorstore.persist()

    print("--- Finished creating vector store ---")
    print(f"Stored {vectorstore._collection.count()} chunks")
    print(f"Vector store saved at: {persist_directory}")

    return vectorstore



def main():

    # LOADING THE FILES
    documents = load_documents(docs_path="docs")

    # CHUNKING THE FILES
    chunks = split_documents(documents)

    # EMBEDDING AND STORING IN VECTOR DB
    vectorstore = create_vector_store(chunks)


    


if __name__ == "__main__":
    main()