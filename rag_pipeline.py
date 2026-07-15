import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import os
import json
import base64
import io
from dotenv import load_dotenv

# OCR
import pytesseract
from PIL import Image

# Unstructured
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# LangChain
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

# -----------------------------
# ⚙️ SET TESSERACT PATH (Windows)
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# 🖼️ OCR FUNCTION
# -----------------------------
def extract_text_from_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""

# -----------------------------
# 📄 LOAD DOCUMENTS
# -----------------------------
def load_documents(folder="docs"):
    elements = []

    print(f"\n📂 Loading documents from: {folder}")

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if os.path.isfile(path):
            print(f"📄 Processing: {file}")

            elems = partition(
                filename=path,
                strategy="fast",
                infer_table_structure=True
            )

            elements.extend(elems)

    print(f"✅ Total elements extracted: {len(elements)}")
    return elements

# -----------------------------
# 🔨 CHUNKING (TITLE-BASED)
# -----------------------------
def chunk_elements(elements):
    print("\n🔨 Chunking documents...")

    chunks = chunk_by_title(
        elements,
        max_characters=2000,
        new_after_n_chars=1500,
        combine_text_under_n_chars=300
    )

    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# -----------------------------
# 🧠 CONTENT SEPARATION
# -----------------------------
def separate_content(chunk):
    data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "image_texts": []
    }

    if hasattr(chunk.metadata, "orig_elements"):
        for el in chunk.metadata.orig_elements:

            # Table handling
            if type(el).__name__ == "Table":
                html = getattr(el.metadata, "text_as_html", el.text)
                data["tables"].append(html)

            # Image handling
            elif type(el).__name__ == "Image":
                if hasattr(el.metadata, "image_base64"):
                    img = el.metadata.image_base64
                    data["images"].append(img)

                    # OCR extraction
                    text = extract_text_from_image(img)
                    if text:
                        data["image_texts"].append(text)

    return data

# -----------------------------
# 🧾 PROCESS CHUNKS
# -----------------------------
def process_chunks(chunks):
    print("\n🧠 Processing chunks...")

    documents = []

    for chunk in chunks:
        data = separate_content(chunk)

        # Build content
        content = data["text"][:500]

        if data["tables"]:
            content += f"\n[Contains {len(data['tables'])} table(s)]"

        if data["image_texts"]:
            content += "\n[Image Text]: " + " ".join(data["image_texts"][:2])

        documents.append(
            Document(
                page_content=content,
                metadata={"source": "multi-doc"}
            )
        )

    print(f"✅ Processed documents: {len(documents)}")
    return documents

# -----------------------------
# 🔮 VECTOR STORE
# -----------------------------
def create_vector_store(documents):
    print("\n🔮 Creating vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="db/chroma_db"
    )

    print("✅ Vector store created")
    return db

# -----------------------------
# 🤖 QUERY FUNCTION
# -----------------------------
def ask_question(db, query):
    print(f"\n👉 Question: {query}")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    print("\n📄 Retrieved Context:\n")
    for i, d in enumerate(docs):
        print(f"Doc {i+1}: {d.page_content[:200]}...\n")

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using ONLY the context below.

Question:
{query}

Context:
{context}

If answer is not present, say:
"I don't have enough information."

Answer:
"""

    llm = ChatOllama(model="phi3")

    response = llm.invoke(prompt)

    print("\n🤖 Answer:\n")
    print(response.content)

# -----------------------------
# 🚀 MAIN
# -----------------------------
def main():
    elements = load_documents("docs")
    chunks = chunk_elements(elements)
    documents = process_chunks(chunks)
    db = create_vector_store(documents)

    print("\n💬 RAG System Ready! Type 'exit' to quit.\n")

    while True:
        query = input("💬 Ask: ")
        if query.lower() == "exit":
            break

        ask_question(db, query)


if __name__ == "__main__":
    main()