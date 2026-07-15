# import pickle
# import networkx as nx
# import matplotlib.pyplot as plt
#
#
# # Load saved knowledge graph
# with open("knowledge_graph.pkl", "rb") as f:
#     graph = pickle.load(f)
#
#
# print("Total Nodes:", len(graph.nodes()))
# print("Total Edges:", len(graph.edges()))
#
#
# # Limit size for readability
# subgraph_nodes = list(graph.nodes())[:30]
#
# subgraph = graph.subgraph(subgraph_nodes)
#
#
# plt.figure(figsize=(12, 10))
#
# pos = nx.spring_layout(subgraph, seed=42)
#
#
# # Draw nodes
# nx.draw_networkx_nodes(
#     subgraph,
#     pos,
#     node_size=700
# )
#
# # Draw edges
# nx.draw_networkx_edges(
#     subgraph,
#     pos
# )
#
# # Draw labels
# nx.draw_networkx_labels(
#     subgraph,
#     pos,
#     font_size=8
# )
#
#
# plt.title("Knowledge Graph Visualization")
#
# plt.axis("off")
#
# plt.show()

import pickle
import networkx as nx
import matplotlib.pyplot as plt


# Load graph
with open("knowledge_graph.pkl", "rb") as f:
    graph = pickle.load(f)


# Choose focus entity
focus_entity = "Microsoft"


# Get neighbors
if focus_entity in graph:

    neighbors = list(graph.neighbors(focus_entity))

    nodes_to_draw = [focus_entity] + neighbors

    subgraph = graph.subgraph(nodes_to_draw)

else:

    print("Entity not found")
    exit()


plt.figure(figsize=(10, 8))

pos = nx.spring_layout(
    subgraph,
    seed=42
)


nx.draw(
    subgraph,
    pos,
    with_labels=True,
    node_size=900,
    font_size=9
)


edge_labels = nx.get_edge_attributes(
    subgraph,
    "relation"
)

nx.draw_networkx_edge_labels(
    subgraph,
    pos,
    edge_labels=edge_labels,
    font_size=8
)


plt.title(f"{focus_entity} Entity Graph")

plt.axis("off")

plt.show()