import pickle

# Load graph
with open("knowledge_graph.pkl", "rb") as f:
    graph = pickle.load(f)

print("\nSample Graph Queries:\n")

# Test queries
test_entities = [
    "Microsoft",
    "NVIDIA",
    "Tesla",
    "Google"
]

for entity in test_entities:

    if entity in graph:

        neighbors = list(graph.neighbors(entity))

        print(f"{entity} → {neighbors}")