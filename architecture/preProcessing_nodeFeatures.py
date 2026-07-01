"""
Utilities for parsing phylogenetic trees in Newick format and converting them
into PyTorch Geometric (PyG) Data objects for Graph Neural Networks.
"""

from __future__ import annotations

from io import StringIO
import networkx as nx
import torch
from Bio import Phylo
from torch_geometric.data import Data

# =========================================================================== #
# BLOCH 1: Pre-processing & Node Features
# =========================================================================== #
def nwk_to_pyg_data(newick_str):

    """
    Parses a Newick tree string, extracts topological features,
    and builds a bidirectional PyG Data object.

    Node Features (X) shape [N, 4]:
        - [:, 0]: Node degree (float)
        - [:, 1]: Is-leaf binary indicator (1.0 if leaf, 0.0 otherwise)
        - [:, 2]: Shortest path distance to the root (float)
        - [:, 3]: Numeric taxonomy node_id (0 for internal nodes or non-numeric leaves)
    """

    # read tree using Biopython
    tree = Phylo.read(StringIO(newick_str), "newick")

    # convert into a graph NetworkX (directional graph)
    net_directed = nx.DiGraph(Phylo.to_networkx(tree))

    ## indentify root (to calculate distance to root)
    ## root - node without parents
    roots = [n for n, d in net_directed.in_degree() if d == 0]
    root = roots[0] if roots else list(net_directed.nodes())[0]

    mapping = {node: i for i, node in enumerate(net_directed.nodes())}
    inverse_mapping = {v: k for k, v in mapping.items()}

    # connectivity matrix (who connects to whom)
    # note: code adds connections in both ways ([u, v] e [v, u])
    edges = []
    for u, v in net_directed.edges():
        edges.append([mapping[u], mapping[v]]) # way A -> B
        edges.append([mapping[v], mapping[u]]) # way B -> A (Bidirectional)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # calculate all distances to root in once
    root_dist = nx.shortest_path_length(net_directed, source=root)

    # identify all leaves from the actual tree
    leaves_names = {t.name for t in tree.get_terminals() if t.name}

    node_features = []
    for i in range(len(mapping)):
        node = inverse_mapping[i]

        # feature 1- node degree
        degree = float(net_directed.degree(node))

        # feature 2 -is leaf
        # if the node has a name and the name is in our terminals list from the tree
        node_name = str(node.name).strip() if node.name else None
        is_leaf = 1.0 if node_name in leaves_names else 0.0

        # feature 3 - distance to root
        d_root = float(root_dist[node])

        node_id = 0
        if is_leaf == 1.0 and node_name and node_name.isdigit():
            node_id = int(node_name)

        node_features.append([degree, is_leaf, d_root, node_id])

    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x= x, edge_index= edge_index)
