"""
Inference script for the Siamese-GIN SPR-distance predictor.
Loads a pre-trained model checkpoint and runs predictions on a target CSV.
"""

from __future__ import annotations

import argparse
import sys
from linecache import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project-local imports
from architecture.newickToData import nwk_to_pyg_data
from architecture.GINArchitecture import SPR_GIN_Predictor
from processTree import NwkCache, PairDataset, pair_collate, resolve_pair_filenames, metrics

def main():
    p = argparse.ArgumentParser(description="Run inference using a trained SPR GIN model.")
    p.add_argument("--csv", required=True, help="Target CSV file with pairs to predict")
    p.add_argument("--repository", required=True, help="Folder containing the .nwk files")
    p.add_argument("--model_weights", required=True, help="Path to the trained 'best_model.pth'")
    p.add_argument("--out-predictions", default="predictions_output.csv", help="Where to save results")

    # Model parameters (Must match the architecture configurations used during training)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--pool", default="add", choices=["add", "mean"])
    p.add_argument("--log-target", action="store_true", help="Set if the model was trained with --log-target")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    repo_path = Path(args.repository)

    # Load and parse the target CSV
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df)} rows from {args.csv}")

    rows, valid_indices = [], []
    for idx, row in df.iterrows():
        try:
            fa, fb = resolve_pair_filenames(row, repo_path)
        except Exception as e:
            print(f"  [!] Row {idx} skipped: {e}")
            continue
        if not fa.exists() or not fb.exists():
            print(f"  [!] Files not found for row {idx}: {fa.name} or {fb.name}")
            continue

        # If the target column exists, we extract it; otherwise, we pass a dummy 0.0
        target_val = float(row["spr_distance"]) if "spr_distance" in row else 0.0
        rows.append((str(fa), str(fb), target_val))
        valid_indices.append(idx)

    if not rows:
        sys.exit("No valid tree pairs matched. Verification of paths or repository required.")

    # Parse and cache Newick structures
    print("Parsing Newick graphs into memory...")
    cache = NwkCache()
    used_paths = sorted({p for triple in rows for p in triple[:2]})
    for path in used_paths:
        cache.get(path)

    # Initialize Dataset and DataLoader
    dataset = PairDataset(rows, cache, log_target=False) # Keep targets raw for evaluation
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=pair_collate)

    # Reconstruct Model Architecture and Load Weights
    print(f"Loading model weights from {args.model_weights}...")

    # carrega o checkpoint primeiro para ler a dimensão real do embedding de treino
    checkpoint = torch.load(args.model_weights, map_location=device)
    if "embed.weight" in checkpoint:
        num_species = checkpoint["embed.weight"].shape[0]  # Vai detetar os 9659 automaticamente
    else:
        # Fallback de segurança caso o formato mude
        max_id = cache.max_node_id()
        num_species = max_id + 1

    print(f"  -> Detected trained num_species parameter = {num_species}")

    model = SPR_GIN_Predictor(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_species=num_species,
        embed_dim=args.embed_dim,
        dropout=0.0,
        pool=args.pool,
    ).to(device)

    # Load state dict safely across device types (CPU/GPU)
    model.load_state_dict(checkpoint) ## carrega pesos de forma segura com o checkpoint
    model.eval()

    # Execute Prediction Generation
    print("Running evaluation...")
    preds, tgts = [], []
    with torch.no_grad():
        for batch_a, batch_b, y in loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)

            yhat = model(batch_a, batch_b)
            if args.log_target:
                yhat = torch.expm1(yhat)
            yhat = yhat.clamp(min=0.0)

            preds.extend(yhat.cpu().tolist())
            tgts.extend(y.cpu().tolist())

    # Save outputs and report metricts if ground truth targets exist
    df_valid = df.iloc[valid_indices].copy()
    df_valid["predicted_spr_distance"] = preds
    df_valid.to_csv(args.out_predictions, index=False)
    print(f"Predictions successfully saved to: {args.out_predictions}")

    if "spr_distance" in df.columns:
        mae, rmse, r2, mape = metrics(np.array(tgts), np.array(preds))
        print("\n" + "=" * 40)
        print("PERFORMANCE METRICS ON TEST FILE")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R2   = {r2:.4f}")
        print(f"  MAPE = {mape:.4f}")
        print("=" * 40)

if __name__ == "__main__":
    main()