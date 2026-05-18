import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_or_generate_graph_data
from config import DATA_CFG, PATHS
import os

def visualize_knowledge_graph():
    graph_data  = get_or_generate_graph_data()
    edge_index  = graph_data["edge_index"]
    edge_type   = graph_data["edge_type"]
    class_names = graph_data["class_names"]
    edge_types  = DATA_CFG.edge_types

    G = nx.DiGraph()
    G.add_nodes_from(range(len(class_names)))

    edge_colors = {
        0: "#E53935",   # exploits    — красный
        1: "#FB8C00",   # precedes    — оранжевый
        2: "#43A047",   # targets     — зелёный
        3: "#1E88E5",   # mitigated_by— синий
        4: "#8E24AA",   # similar_to  — фиолетовый
    }

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        etype = edge_type[i].item()
        G.add_edge(src, dst, etype=etype, color=edge_colors[etype])

    plt.figure(figsize=(14, 10))
    pos    = nx.spring_layout(G, seed=42, k=2.5)
    labels = {i: name.replace("_", "\n") for i, name in enumerate(class_names)}

    # Узлы
    nx.draw_networkx_nodes(G, pos, node_size=2000,
                           node_color="#37474F", alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=7, font_color="white",
                            font_weight="bold")

    # Рёбра по типам
    for etype_idx, etype_name in enumerate(edge_types):
        edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d["etype"] == etype_idx]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges,
                edge_color=edge_colors[etype_idx],
                arrows=True, arrowsize=15,
                width=1.5, alpha=0.7,
                connectionstyle="arc3,rad=0.1",
                label=etype_name,
            )

    plt.legend(
        handles=[
            plt.Line2D([0],[0], color=c, linewidth=2, label=edge_types[i])
            for i, c in edge_colors.items()
        ],
        loc="upper left", fontsize=9,
    )
    plt.title("CyberThreat Knowledge Graph\n"
              "Semantic Relations between Attack Types",
              fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(PATHS["plots_dir"], "knowledge_graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Knowledge Graph saved -> {out}")

if __name__ == "__main__":
    visualize_knowledge_graph()