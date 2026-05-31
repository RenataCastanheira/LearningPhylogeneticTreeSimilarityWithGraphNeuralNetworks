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

# --------------------------------------------------------------------------- #
# Newick to PyG Data Converter
# --------------------------------------------------------------------------- #
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

    # ler árvore usando Biopython
    tree = Phylo.read(StringIO(newick_str), "newick")

    # converter para um grafo NetworkX # FORÇAR GRAFO DIRECIONADO
    net_directed = nx.DiGraph(Phylo.to_networkx(tree)) #Phylo.to_networkx(tree)

    ## identificar a raiz (importante para a distância à raiz)
    ## rais - nó sem pais
    roots = [n for n, d in net_directed.in_degree() if d == 0]
    root = roots[0] if roots else list(net_directed.nodes())[0]

    mapping = {node: i for i, node in enumerate(net_directed.nodes())}
    inverse_mapping = {v: k for k, v in mapping.items()}

    # criar o edge_index (quem se liga a quem)
    # matriz de conectividade -- lista de pares que diz "o nó 0 está ligado ao nó 1"
    # nota: o código adiciona ligações nos dois sentidos ([u, v] e [v, u])
    edges = []
    for u, v in net_directed.edges():
        edges.append([mapping[u], mapping[v]]) # sentido A -> B
        edges.append([mapping[v], mapping[u]]) # sentido B -> A (Bidirecional)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # calcular todas as distancias à raiz de uma vez
    root_dist = nx.shortest_path_length(net_directed, source=root)

    # primeiro identificamos todas as folhas da árvore atual
    # o Biopython guarda as folhas no método get_terminals()
    leaves_names = {t.name for t in tree.get_terminals() if t.name}

    # criar as features dos nós (X)
    # usamos o grau do nó como feature inicial
    node_features = []
    for i in range(len(mapping)):
        node = inverse_mapping[i]

        # feature 1- grau do nó
        degree = float(net_directed.degree(node))

        # feature 2
        # se o nó tem um nome e esse nome está na nossa lista de terminais da própria árvore
        node_name = str(node.name).strip() if node.name else None
        is_leaf = 1.0 if node_name in leaves_names else 0.0

        # feature 3 : distancia à raiz
        d_root = float(root_dist[node])

        node_id = 0
        if is_leaf == 1.0 and node_name and node_name.isdigit():
            node_id = int(node_name)

        node_features.append([degree, is_leaf, d_root, node_id])

    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x= x, edge_index= edge_index)

