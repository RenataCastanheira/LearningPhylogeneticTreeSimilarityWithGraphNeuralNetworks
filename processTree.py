"""
Train and evaluate the Siamese-GIN SPR-distance predictor.

This is a revised version of the original `processTree.py`. It fixes the
methodological problems found in the BetaVersionG18 review and in the 300-epoch
log analysis:

  - Proper train / val / test split (with a fixed seed).
  - Caches each parsed .nwk -> PyG Data object once (instead of re-parsing every
    epoch, which made the original loop O(num_epochs x N x parse_time)).
  - Batches pairs via a PyG `Batch.from_data_list` collate, instead of training
    one example at a time.
  - Sizes `num_species` dynamically from the maximum node_id actually present
    in the data (no more silent `IndexError`s being swallowed by a broad
    except). Default >= max_id + 1.
  - Adds weight decay, dropout, Huber (SmoothL1) loss, ReduceLROnPlateau, and
    early stopping on validation MAE.
  - Optional `--log-target` flag: trains on log1p(SPR) and inverse-transforms
    at evaluation time. Helps with the long-tail target distribution we
    observed (median 21, mean 127, max 900).
  - Reports baselines (predict-train-mean and predict-train-median) alongside
    the model's MAE/RMSE/R^2/MAPE so the result is interpretable.
  - Saves the best weights, the training curve and the test predictions to
    --out-dir.

Usage:
    python processTree.py \\
        --csv      /path/to/spr_metrics.csv \\
        --repository /path/to/repository \\
        --out-dir  runs/spr_gnn_full \\
        --num-epochs 300 --batch-size 16 --patience 25

Expected CSV columns:
    specie, category, size, type, spr_distance
Optional (for mixed-shuffle pairs like "NJ vs NJ_sh"):
    sh_a, sh_b   (booleans / 0-1)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

# Project-local
from architecture.newickToData import nwk_to_pyg_data
from architecture.GINArchitecture import SPR_GIN_Predictor


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# CSV row -> two .nwk paths
# --------------------------------------------------------------------------- #
def resolve_pair_filenames(row, repo_path: Path) -> tuple[Path, Path]:
    """
    Build the two .nwk paths for one CSV row.

    Directly uses 'pathA'/'patha' and 'pathB'/'pathb' columns from the CSV,
    ensuring compatibility by removing any redundant 'repository/' prefix.
    """
    # Procura pela coluna tanto em maiúsculas como em minúsculas
    col_a = "pathA" if "pathA" in row else "patha"
    col_b = "pathB" if "pathB" in row else "pathb"

    p_a = str(row[col_a]).strip()
    p_b = str(row[col_b]).strip()

    # Se o CSV já traz "repository/nome.nwk", removemos "repository/"
    # para evitar caminhos duplicados como "repository/repository/nome.nwk"
    if p_a.startswith("repository/"):
        p_a = p_a.replace("repository/", "", 1)
    if p_b.startswith("repository/"):
        p_b = p_b.replace("repository/", "", 1)

    # Combina o repo_path (passado por argumento) com o nome limpo do ficheiro
    return repo_path / p_a, repo_path / p_b

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


# --------------------------------------------------------------------------- #
# Evaluation (always in the original SPR scale)
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device, log_target: bool):
    """Run the model on `loader`, return (true_y, predicted_y) in raw scale."""
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch_a, batch_b, y in loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            yhat = model(batch_a, batch_b)
            yhat = torch.expm1(yhat) if log_target else yhat
            yhat = yhat.clamp(min=0.0)         # SPR distance is non-negative
            preds.extend(yhat.cpu().tolist())
            tgts.extend(y.cpu().tolist())     # `y` already in raw scale here
    return np.asarray(tgts, dtype=float), np.asarray(preds, dtype=float)


def metrics(t: np.ndarray, p: np.ndarray):
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    r2 = float(r2_score(t, p)) if len(t) > 1 else float("nan")
    nz = t > 0
    mape = (
        float(mean_absolute_percentage_error(t[nz], p[nz])) if nz.any() else float("nan")
    )
    return mae, rmse, r2, mape


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Train Siamese GIN SPR predictor.")
    p.add_argument("--csv", required=True,
                   help="C"
                        "SV with columns specie,category,size,type,spr_distance [+ sh_a,sh_b]")
    p.add_argument("--repository", required=True, help="Folder with the .nwk files")
    p.add_argument("--out-dir", default="runs/spr_gnn",
                   help="Where to save weights, training curve, predictions")
    # training
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25,
                   help="Early-stop patience on val MAE (epochs).")
    p.add_argument("--lr-patience", type=int, default=10,
                   help="ReduceLROnPlateau patience (epochs).")
    # model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool", default="add", choices=["add", "mean"])
    # data
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-target", action="store_true",
                   help="Train on log1p(SPR); evaluate in original scale.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    repo_path = Path(args.repository)
    device = torch.device(args.device)
    print(f"Device: {device}\nSeed:   {args.seed}\nOutDir: {out_dir.resolve()}\n")

    # ----- 1. Load CSV and resolve file paths ------------------------------ #
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lower()
    print(f"CSV: {len(df)} rows from {args.csv}")

    rows, missing = [], 0
    for _, row in df.iterrows():
        try:
            fa, fb = resolve_pair_filenames(row, repo_path)
        except Exception as e:
            print(f"  [!] bad row skipped: {e}")
            missing += 1
            continue
        if not fa.exists() or not fb.exists():
            missing += 1
            continue
        rows.append((str(fa), str(fb), float(row["spr_distance"])))
    print(f"Resolved {len(rows)} valid pairs ({missing} skipped)\n")
    if not rows:
        sys.exit("No valid pairs to train on. Check --csv and --repository paths.")

    # ----- 2. Train / val / test split (random, seeded) -------------------- #
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(rows))
    n_test = int(round(args.test_frac * len(rows)))
    n_val = int(round(args.val_frac * len(rows)))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    print(f"Split: {len(train_rows)} train | {len(val_rows)} val | {len(test_rows)} test\n")

    # ----- 3. Cache all .nwk files; size num_species dynamically ----------- #
    print("Parsing and caching .nwk -> PyG Data ...")
    cache = NwkCache()
    used = sorted({p for triple in rows for p in triple[:2]})
    for i, path in enumerate(used, 1):
        try:
            cache.get(path)
        except Exception as e:
            sys.exit(f"  Failed to parse {path}: {e}")
        if i % 50 == 0 or i == len(used):
            print(f"  cached {i}/{len(used)}")
    max_id = cache.max_node_id()
    num_species = max_id + 1
    print(f"\nMax node_id observed = {max_id}  ->  num_species = {num_species}\n")

    # ----- 4. Datasets & loaders ------------------------------------------ #
    train_ds = PairDataset(train_rows, cache, log_target=args.log_target)
    val_ds   = PairDataset(val_rows,   cache, log_target=False)
    test_ds  = PairDataset(test_rows,  cache, log_target=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=pair_collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=pair_collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=pair_collate)

    # ----- 5. Model / optimiser / scheduler / loss ------------------------- #
    model = SPR_GIN_Predictor(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_species=num_species,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)

    optimiser = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode="min",
                                  factor=0.5, patience=args.lr_patience)
    # Huber: robust to the long tail (mean=127, max=900) we saw in the data.
    criterion = nn.SmoothL1Loss()

    # ----- 6. Train with early stopping ----------------------------------- #
    history, best_val_mae, best_epoch, since_best = [], float("inf"), 0, 0
    best_path = out_dir / "best_model.pth"
    print(f"Training up to {args.num_epochs} epochs (patience={args.patience}).\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch_a, batch_b, y in train_loader:
            batch_a = batch_a.to(device); batch_b = batch_b.to(device)
            y_target = y.to(device)
            optimiser.zero_grad()
            yhat = model(batch_a, batch_b)
            loss = criterion(yhat, y_target)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))

        v_t, v_p = evaluate(model, val_loader, device, args.log_target)
        val_mae, val_rmse, val_r2, val_mape = metrics(v_t, v_p)
        train_loss = float(np.mean(losses))
        lr_now = optimiser.param_groups[0]["lr"]
        scheduler.step(val_mae)

        history.append(dict(
            epoch=epoch, train_loss=train_loss, lr=lr_now,
            val_mae=val_mae, val_rmse=val_rmse, val_r2=val_r2, val_mape=val_mape,
        ))

        improved = val_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae, best_epoch, since_best = val_mae, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            since_best += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            tag = " *" if improved else ""
            print(f"  epoch {epoch:3d} | train_loss {train_loss:7.3f} | "
                  f"val MAE {val_mae:7.2f}  R2 {val_r2:6.3f} | lr {lr_now:.1e}{tag}")

        if since_best >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no val MAE improvement for {args.patience} epochs).")
            break

    print(f"\nBest val MAE = {best_val_mae:.3f} at epoch {best_epoch}\n")

    # ----- 7. Final test with the best weights ---------------------------- #
    model.load_state_dict(torch.load(best_path, map_location=device))
    t_t, t_p = evaluate(model, test_loader, device, args.log_target)
    mae, rmse, r2, mape = metrics(t_t, t_p)

    # Baselines for context
    train_targets = np.array([r[2] for r in train_rows])
    mean_pred = float(np.mean(train_targets))
    median_pred = float(np.median(train_targets))
    base_mean_mae = float(np.mean(np.abs(t_t - mean_pred)))
    base_med_mae = float(np.mean(np.abs(t_t - median_pred)))

    print("=" * 56)
    print("FINAL TEST RESULTS")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R2   = {r2:.4f}")
    print(f"  MAPE = {mape:.4f}")
    print(f"  baseline 'predict train mean'   ({mean_pred:7.2f}) -> MAE {base_mean_mae:.2f}")
    print(f"  baseline 'predict train median' ({median_pred:7.2f}) -> MAE {base_med_mae:.2f}")
    print("=" * 56)

    # Save artefacts
    pd.DataFrame(history).to_csv(out_dir / "training_curve.csv", index=False)
    pd.DataFrame({"real": t_t, "pred": t_p}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )
    with open(out_dir / "summary.txt", "w") as f:
        f.write(
            f"Best val MAE: {best_val_mae:.4f} (epoch {best_epoch})\n"
            f"Test  MAE: {mae:.4f}\n"
            f"Test  RMSE: {rmse:.4f}\n"
            f"Test  R2:  {r2:.4f}\n"
            f"Test  MAPE: {mape:.4f}\n"
            f"Baseline mean-MAE:   {base_mean_mae:.4f}\n"
            f"Baseline median-MAE: {base_med_mae:.4f}\n"
            f"num_species: {num_species}\n"
            f"hidden_dim: {args.hidden_dim}\n"
            f"dropout: {args.dropout}\n"
            f"pool: {args.pool}\n"
            f"log_target: {args.log_target}\n"
            f"seed: {args.seed}\n"
        )
    print(f"\nArtefacts saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
