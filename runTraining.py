import os
import sys
import argparse
import shutil
from pathlib import Path


tool_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory_gnn = os.path.dirname(tool_directory)

if tool_directory not in sys.path:
    sys.path.insert(0, tool_directory)
if parent_directory_gnn not in sys.path:
    sys.path.insert(0, parent_directory_gnn)


from processTree import main as run_training
from settings.utils_pipeline import download_and_extract_all, validate_and_filter_csv


def main():
    p = argparse.ArgumentParser(description="Pipeline Automático de Treino da GNN (3 Ficheiros CSV)")
    # Independent CSV paths
    p.add_argument("--train-csv", default="scripts/train_gnn.csv", help="Caminho para o CSV de treino")
    p.add_argument("--val-csv", default="scripts/validation_gnn.csv", help="Caminho para o CSV de validação")
    p.add_argument("--test-csv", default="scripts/test_gnn.csv", help="Caminho para o CSV de teste")
    p.add_argument("--out-dir", default="runs/spr_gnn_full_3hidden", help="Onde guardar os pesos e curvas de treino")

    # Train Hyperparameters
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr-patience", type=int, default=10)

    # GIN model configurations
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool", default="add", choices=["add", "mean"])
    p.add_argument("--log-target", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    base_path = Path(".").resolve()
    target_repo = base_path / "repository"

    print("\n=== STEP 1: DATASET & DIRECTORIES ===")
    # Calls centralized function utils_pipeline
    download_and_extract_all(target_repo)

    num_trees = len([f for f in target_repo.glob("*.nwk") if not f.name.startswith("._")])
    if num_trees == 0:
        sys.exit("\n[Error] No valid tree was found in the 'repository' folder.")

    # Filters and validates the 3 CSV files independently using the modular utility.
    filtered_train_csv = validate_and_filter_csv(Path(args.train_csv), base_path, "treino")
    validated_csv_path = validate_and_filter_csv(Path(args.val_csv), base_path, "validação")
    validated_test_csv = validate_and_filter_csv(Path(args.test_csv), base_path, "teste")

    print(f"\n=== STEP 2: GNN TRAINING EXECUTION ({num_trees} mapped trees) ===")

    # Sets up the simulated command-line arguments for processTree.py
    sys.argv = [
        "processTree.py",
        "--train-csv", str(filtered_train_csv),
        "--val-csv", str(validated_csv_path),
        "--test-csv", str(validated_test_csv),
        "--zenodo_scripts", str(target_repo),
        "--repository", str(target_repo),
        "--out-dir", args.out_dir,
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--lr-patience", str(args.lr_patience),  # <- CORRIGIDO AQUI
        "--hidden-dim", str(args.hidden_dim),
        "--embed-dim", str(args.embed_dim),
        "--dropout", str(args.dropout),
        "--pool", args.pool,
        "--seed", str(args.seed)
    ]

    if args.log_target:
        sys.argv.append("--log-target")

    try:
        run_training()
    finally:
        # Safe and thorough cleanup of the 3 temporary CSV files created
        if filtered_train_csv != Path(args.train_csv) and filtered_train_csv.exists():
            os.remove(filtered_train_csv)
        if validated_csv_path != Path(args.val_csv) and validated_csv_path.exists():
            os.remove(validated_csv_path)
        if validated_test_csv != Path(args.test_csv) and validated_test_csv.exists():
            os.remove(validated_test_csv)

    print("\n=== TRAINING PROCESS COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()