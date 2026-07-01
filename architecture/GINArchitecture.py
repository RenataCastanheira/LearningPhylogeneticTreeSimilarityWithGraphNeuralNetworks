"""
Siamese Graph Isomorphism Network (GIN) regressor for SPR distance.

Changes vs. the original BetaVersion:
  - Removed the accidental `from bdb import effective` IDE import.
  - Added optional dropout in the regression head (default 0.3).
  - Added a `pool` switch ("add" | "mean") so size-sensitivity can be tested.
  - Squeezed the final scalar to shape (batch,) instead of (batch, 1) so the
    loss broadcasting is unambiguous.
  - Architecture is backward-compatible: `SPR_GIN_Predictor(input_dim=4,
    hidden_dim=128)` gives the same network as before when dropout=0, pool="add".
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout

from architecture.output_MLPRegressionHead import output_and_MLP_Regression

# =========================================================================== #
# BLOCH 2: GIN (Graph Isomorphism Network) Module
# =========================================================================== #
class SPR_GIN_Predictor(torch.nn.Module):
    """
    Processes two phylogenetic trees with shared weights, pools each to a
    graph embedding, concatenates, and regresses to a scalar SPR distance.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_species: int = 10_000,
        embed_dim: int = 16,
        dropout: float = 0.3,
        pool: str = "add",
    ):
        super().__init__()

        # -- Embedding layer [6, 18] --
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(num_species, embed_dim)

        # 3 continuous features (degree, is_leaf, d_root) + embedding for node_id
        effective_input = (input_dim - 1) + embed_dim  # = 19 with the defaults

        # -- GIN Convolution Layers (sharing weights) --
        # LAYER 1
        nn1 = Sequential(
            Linear(effective_input, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden_dim)

        # LAYER 2
        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(hidden_dim)

        # LAYER 3
        nn3 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # -- Global Pooling --
        if pool == "add":
            self._pool = global_add_pool
        elif pool == "mean":
            self._pool = global_mean_pool
        else:
            raise ValueError(f"Unknown pool {pool!r}; choose 'add' or 'mean'.")

        output_and_MLP_Regression(self, hidden_dim, dropout)

    # -------------------------------------------------------------------- #
    # Internal
    def _embed_and_process(self, data):
        x_cont = data.x[:, :3]                    # degree, is_leaf, d_root
        x_ids = data.x[:, 3].long()               # node_id
        x_emb = self.embed(x_ids)                 # (N, embed_dim)
        x = torch.cat([x_cont, x_emb], dim=1)     # (N, 3 + embed_dim)

        x = F.relu(self.bn1(self.conv1(x, data.edge_index)))
        x = F.relu(self.bn2(self.conv2(x, data.edge_index)))
        x = F.relu(self.bn3(self.conv3(x, data.edge_index)))

        batch = data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self._pool(x, batch)

    # -------------------------------------------------------------------- #
    def forward(self, data_a, data_b):
        v_a = self._embed_and_process(data_a)
        v_b = self._embed_and_process(data_b)
        combined = torch.cat([v_a, v_b], dim=1)
        return self.fc(combined).squeeze(-1)      # shape (batch,)