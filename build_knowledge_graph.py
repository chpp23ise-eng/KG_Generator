from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import networkx as nx
import pickle


# Load embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector store
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)


# Get all stored documents
# Get documents from vector DB safely
retriever = db.as_retriever(
    search_kwargs={"k": 20}
)

retrieved_docs = retriever.invoke(
    "Extract all important entities and relationships"
)

documents = retrieved_docs


# Load local LLM
llm = OllamaLLM(
    model="mistral"
)


# Create transformer
graph_transformer = LLMGraphTransformer(
    llm=llm,
    strict_mode=False
)


print("Building knowledge graph...")


# Convert documents → graph documents
graph_docs = []

for doc in documents:

    try:
        result = graph_transformer.convert_to_graph_documents(
            [doc]  # process ONE document
        )

        graph_docs.extend(result)

    except Exception as e:
        print("Skipping problematic document...")
        print(e)



# Debug print
print("\nSample relationships:\n")

if len(graph_docs) > 0:

    for rel in graph_docs[0].relationships:
        print(rel)

# Create NetworkX graph
graph = nx.Graph()


print("Sample relationships:")



for doc in graph_docs:

    for rel in doc.relationships:

        # Fix list outputs from Mistral
        source = rel.source.id
        target = rel.target.id

        if isinstance(source, list):
            source = source[0]

        if isinstance(target, list):
            target = target[0]

        source = str(source)
        target = str(target)

        relation = str(rel.type)

        graph.add_edge(
            source,
            target,
            relation=relation
        )


# Save graph
with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump(graph, f)


print("\nKnowledge graph created!")
print(f"Nodes: {len(graph.nodes())}")
print(f"Edges: {len(graph.edges())}")