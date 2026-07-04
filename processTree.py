"""
Train and evaluate the Siamese-GIN SPR-distance predictor using strict CSV splits.

Usage:
    python processTree.py \
        --train-csv /path/to/train.csv \
        --val-csv   /path/to/val.csv \
        --test-csv  /path/to/test.csv \
        --repository /path/to/repository \
        --out-dir  runs/spr_gnn_full \
        --num-epochs 300 --batch-size 16 --patience 25
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

# Imports do teu projeto organizados
from architecture.GINArchitecture import SPR_GIN_Predictor
from utils.data_pipeline import NwkCache, PairDataset, pair_collate
from utils.metrics_utils import evaluate, metrics


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
    """
    # Procura flexível pelas colunas já convertidas em minúsculas
    col_a = "patha" if "patha" in row else ("tree_1" if "tree_1" in row else "pathA")
    col_b = "pathb" if "pathb" in row else ("tree_2" if "tree_2" in row else "pathB")

    p_a = str(row[col_a]).strip()
    p_b = str(row[col_b]).strip()

    # Remove qualquer prefixo redundante que possa vir no CSV para evitar caminhos duplicados
    for prefix in ["zenodo_scripts/", "repository/"]:
        if p_a.startswith(prefix):
            p_a = p_a.replace(prefix, "", 1)
        if p_b.startswith(prefix):
            p_b = p_b.replace(prefix, "", 1)

    return repo_path / p_a, repo_path / p_b


def process_single_csv(csv_path: Path, repo_path: Path, name: str) -> list:
    """
    Lê um ficheiro CSV específico e resolve os caminhos físicos das árvores.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    if 'patha' not in df.columns and 'tree_1' in df.columns:
        df = df.rename(columns={'tree_1': 'patha', 'tree_2': 'pathb'})

    print(f"CSV {name}: {len(df)} rows from {csv_path}")

    rows, missing = [], 0
    for _, row in df.iterrows():
        try:
            fa, fb = resolve_pair_filenames(row, repo_path)
        except Exception as e:
            print(f"  [!] bad row skipped in {name}: {e}")
            missing += 1
            continue

        # Verificação robusta: se não encontrar com .nwk, tenta ver se o ficheiro físico existe
        if not fa.exists() or not fb.exists():
            missing += 1
            continue

        rows.append((str(fa), str(fb), float(row["spr_distance"])))

    print(f"  -> Resolved {len(rows)} valid pairs ({missing} skipped) for {name}.\n")
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Train Siamese GIN SPR predictor with 3 distinct CSVs.")
    p.add_argument("--train-csv", required=True, help="CSV containing training pairs")
    p.add_argument("--val-csv", required=True, help="CSV containing validation pairs")
    p.add_argument("--test-csv", required=True, help="CSV containing test pairs")
    p.add_argument("--repository", required=True, help="Folder containing the .nwk files (repository)")
    p.add_argument("--zenodo_scripts", help="Legacy duplicate argument for backwards compatibility")
    p.add_argument("--out-dir", default="runs/spr_gnn", help="Where to save weights, training curve, predictions")

    # training
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25, help="Early-stop patience on val MAE (epochs).")
    p.add_argument("--lr-patience", type=int, default=10, help="ReduceLROnPlateau patience (epochs).")

    # model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool", default="add", choices=["add", "mean"])

    # data configurations
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-target", action="store_true", help="Train on log1p(SPR); evaluate in original scale.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(args.repository if args.repository else args.zenodo_scripts)
    device = torch.device(args.device)
    print(f"Device: {device}\nSeed:   {args.seed}\nOutDir: {out_dir.resolve()}\n")

    # ----- 1. Load CSVs and resolve file paths independently ---------------- #
    train_rows = process_single_csv(Path(args.train_csv), repo_path, "TRAIN")
    val_rows = process_single_csv(Path(args.val_csv), repo_path, "VALIDATION")
    test_rows = process_single_csv(Path(args.test_csv), repo_path, "TEST")

    all_loaded_rows = train_rows + val_rows + test_rows
    if not all_loaded_rows:
        sys.exit("No valid pairs found across any CSV. Check paths and files.")

    print(f"Strict Splits: {len(train_rows)} train | {len(val_rows)} val | {len(test_rows)} test\n")

    # ----- 2. Cache all .nwk files; size num_species dynamically ----------- #
    print("Parsing and caching .nwk -> PyG Data ...")
    cache = NwkCache()
    used = sorted({p for triple in all_loaded_rows for p in triple[:2]})
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

    # ----- 3. Datasets & loaders ------------------------------------------ #
    train_ds = PairDataset(train_rows, cache, log_target=args.log_target)
    val_ds   = PairDataset(val_rows,   cache, log_target=False)
    test_ds  = PairDataset(test_rows,  cache, log_target=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=pair_collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=pair_collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=pair_collate)

    # ----- 4. Model / optimiser / scheduler / loss ------------------------- #
    model = SPR_GIN_Predictor(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_species=num_species,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)

    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=args.lr_patience)
    criterion = nn.SmoothL1Loss()

    # ----- 5. Train with early stopping ----------------------------------- #
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

    # ----- 6. Final test with the best weights ---------------------------- #
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
    pd.DataFrame({"real": t_t, "pred": t_p}).to_csv(out_dir / "test_predictions.csv", index=False)

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