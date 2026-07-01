
import math
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from architecture.preProcessing_nodeFeatures import nwk_to_pyg_data

# --------------------------------------------------------------------------- #
# Cache: parse each .nwk file at most once
# --------------------------------------------------------------------------- #
class NwkCache:
    def __init__(self) -> None:
        self._cache: dict[str, "Data"] = {}

    def get(self, path: str):
        path = str(path)
        if path not in self._cache:
            with open(path) as f:
                self._cache[path] = nwk_to_pyg_data(f.read())
        return self._cache[path]

    def max_node_id(self) -> int:
        mx = 0
        for d in self._cache.values():
            ids = d.x[:, 3].long()
            if ids.numel():
                mx = max(mx, int(ids.max()))
        return mx


# --------------------------------------------------------------------------- #
# Pair dataset and batch collation
# --------------------------------------------------------------------------- #
class PairDataset(Dataset):
    """Holds (path_a, path_b, target) triples; materialises via NwkCache."""

    def __init__(self, rows, cache: NwkCache, log_target: bool = False):
        self.rows = rows
        self.cache = cache
        self.log_target = log_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        file_a, file_b, y = self.rows[idx]
        data_a = self.cache.get(file_a)
        data_b = self.cache.get(file_b)
        if self.log_target:
            y = math.log1p(max(float(y), 0.0))
        return data_a, data_b, float(y)


def pair_collate(batch):
    """Batch the two graph streams independently via PyG Batch."""
    a_list, b_list, ys = zip(*batch)
    return (
        Batch.from_data_list(list(a_list)),
        Batch.from_data_list(list(b_list)),
        torch.tensor(ys, dtype=torch.float),
    )