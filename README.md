# Hybrid GraphRAG System with Knowledge Graph-Enhanced Document Question Answering

## Overview

This project implements a **Hybrid GraphRAG** system that combines semantic vector retrieval with Knowledge Graph-based retrieval to provide accurate, grounded, and explainable answers from uploaded PDF and TXT documents.

Unlike traditional RAG systems that rely only on semantic similarity, this project automatically extracts entities and relationships from documents, constructs a Knowledge Graph, and enriches retrieval using graph traversal before generating responses with a local Large Language Model (LLM).

---

## Features

- PDF and TXT document ingestion
- OCR support for scanned document images (Tesseract OCR)
- Title-based semantic chunking
- Hugging Face sentence embeddings
- ChromaDB vector database
- Maximal Marginal Relevance (MMR) retrieval
- Automatic entity and relationship extraction using LLMGraphTransformer
- Knowledge Graph construction using NetworkX
- One-hop Knowledge Graph traversal for graph context retrieval
- Hybrid retrieval (Vector + Knowledge Graph)
- Local LLM inference using Mistral/Phi-3 via Ollama
- Knowledge Graph visualization

---

## Project Architecture

```text
                 PDF / TXT
                     │
                     ▼
            Document Loader
                     │
                     ▼
          Semantic Chunking
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
Generate Embeddings         Entity & Relationship Extraction
      │                             │
      ▼                             ▼
 ChromaDB                     Knowledge Graph
      │                             │
      │                             ▼
      │                    NetworkX Graph
      │                             │
      └──────────────┬──────────────┘
                     ▼
                User Query
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  Vector Retrieval         Graph Traversal
      (MMR)               (One-hop Neighbors)
         │                       │
         └───────────┬───────────┘
                     ▼
             Hybrid Context
                     ▼
          Local LLM (Mistral/Phi-3)
                     ▼
               Generated Answer
```

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Framework | LangChain |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Knowledge Graph | NetworkX |
| Graph Extraction | LLMGraphTransformer |
| LLM | Mistral / Phi-3 (Ollama) |
| OCR | Tesseract OCR |
| Query Processing | spaCy |
| Graph Visualization | Matplotlib |

---

## Workflow

### Document Processing

- Upload PDF/TXT documents
- Parse document content
- Extract text using OCR (if required)
- Perform semantic chunking

### Vector Database Pipeline

- Generate embeddings using Hugging Face
- Store embeddings in ChromaDB
- Retrieve relevant chunks using MMR

### Knowledge Graph Pipeline

- Extract entities and relationships using LLMGraphTransformer
- Construct Knowledge Graph using NetworkX
- Save graph as `knowledge_graph.pkl`

### Query Processing

- Extract entities from user query
- Retrieve relevant chunks from ChromaDB
- Perform one-hop graph traversal
- Merge vector context and graph context
- Generate final response using the local LLM

---

## Project Structure

```text
HybridGraphRAG/
│
├── docs/
│   ├── sample.pdf
│   └── sample.txt
│
├── chroma_db/
│
├── knowledge_graph.pkl
│
├── ingestion_pipeline.py
├── build_knowledge_graph.py
├── retrieval_pipeline.py
├── query_graph.py
├── app.py
│
├── requirements.txt
├── README.md
└── setup.py
```

---

## How It Works

### 1. Entity Extraction

The uploaded document is divided into semantic chunks.

Each chunk is processed by **LLMGraphTransformer** using a locally running Mistral model to identify important entities.

Example:

```
Microsoft acquired GitHub.
```

Extracted entities:

- Microsoft
- GitHub

---

### 2. Relationship Extraction

The LLM identifies semantic relationships.

Example:

```
Microsoft
      │
 ACQUIRED
      │
 GitHub
```

---

### 3. Knowledge Graph Construction

Entities become nodes.

Relationships become edges.

Example:

```
Microsoft ----ACQUIRED---- GitHub
```

The graph is stored using NetworkX.

---

### 4. Hybrid Retrieval

When the user submits a question:

- ChromaDB retrieves semantically relevant chunks.
- The query entities are matched against the Knowledge Graph.
- Neighboring nodes are retrieved through one-hop graph traversal.
- Vector context and graph context are merged.

---

### 5. Answer Generation

The hybrid context is supplied to the local LLM through Ollama.

The LLM generates a grounded answer based solely on the uploaded documents.

---

## Example Query

**Question**

```
When was Microsoft established?
```

Retrieved Vector Context

```
Relevant document chunks
```

Retrieved Graph Context

```
Microsoft
GitHub
Windows
Microsoft Teams
```

Final Answer

```
Microsoft was founded in 1975...
```

---

## Advantages

- Reduces hallucinations by grounding responses in uploaded documents.
- Uses both semantic similarity and structured entity relationships.
- Supports explainable retrieval using Knowledge Graph visualization.
- Runs completely offline using local LLMs.
- Easily extendable to multiple domains.

---

## Current Limitations

- Performs one-hop graph traversal only.
- Does not generate explicit multi-hop reasoning paths.
- Relationship labels are stored but not yet incorporated into graph retrieval.
- Designed primarily for PDF and TXT documents.

---

## Future Work

- Multi-hop Knowledge Graph reasoning.
- Relationship-aware graph traversal.
- Adaptive retrieval strategy.
- Bloom's Taxonomy-based Question Paper Generation.
- Automatic answer evaluation.
- Support for multimodal documents using Vision-Language Models.

---

## Authors

Developed as a Final Year Engineering Project.

```

---

### ⭐ This README is **GitHub-ready** and accurately reflects your implementation. It doesn't overstate your work, uses correct terminology (Hybrid GraphRAG, one-hop traversal, hybrid retrieval), and is suitable for recruiters, interviewers, and project evaluators.
