import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Garante que as diretorias do projeto estão no sys.path
diretoria_tool = os.path.dirname(os.path.abspath(__file__))
diretoria_pai_gnn = os.path.dirname(diretoria_tool)

if diretoria_tool not in sys.path:
    sys.path.insert(0, diretoria_tool)
if diretoria_pai_gnn not in sys.path:
    sys.path.insert(0, diretoria_pai_gnn)

# Importações do teu projeto e do ficheiro utilitário
from predict import main as run_prediction
from settings.utils_pipeline import download_and_extract_all

def filtrar_csv_por_ficheiros_existentes(caminho_csv: Path, base_dir: Path):
    """
    Lê o CSV de entrada, verifica se os ficheiros de árvores referenciados existem no disco e
    gera um CSV temporário contendo apenas as linhas cujos ficheiros existem no repositório.
    """
    print(f"[Pipeline] Verifying CSV consistency '{caminho_csv.name}'...")

    if not caminho_csv.exists():
        return caminho_csv

    try:
        df = pd.read_csv(caminho_csv)
    except Exception as e:
        print(f"  [!] Error reading CSV: {e}")
        return caminho_csv

    # Mapeia as colunas exatas do seu CSV: pathA e pathB
    colunas_arvore = [col for col in df.columns if col.strip() in ['pathA', 'pathB', 'tree_1', 'tree_2']]
    if not colunas_arvore:
        print("  [!] Tree Columns non identified in CSV. Passing original CSV...")
        return caminho_csv

    linhas_validas = []

    for idx, row in df.iterrows():
        par_valido = True
        for col in colunas_arvore:
            caminho_relativo = str(row[col]).strip()

            # Reconstrói o caminho completo juntando a diretoria base ao que está no CSV
            ficheiro_alvo = base_dir / caminho_relativo

            # Trata a variante com ou sem SH
            variante_sh = Path(
                str(ficheiro_alvo).replace("_upgma.nwk", "_sh_upgma.nwk").replace("_nj.nwk", "_sh_nj.nwk"))

            if not ficheiro_alvo.exists() and not variante_sh.exists():
                par_valido = False
                break

        if par_valido:
            linhas_validas.append(row)

    if len(linhas_validas) == 0:
        print("  [!] Warning: No line filtered manually, sending original to predict.py.")
        return caminho_csv

    df_filtrado = pd.DataFrame(linhas_validas)
    csv_temporario = caminho_csv.parent / f"temp_filtrado_{caminho_csv.name}"
    df_filtrado.to_csv(csv_temporario, index=False)

    print(f"  [+] CSV Validated! {len(df_filtrado)} lines ready to be processed.")
    return csv_temporario


def main():
    p = argparse.ArgumentParser(description="Automatic Pipeline GNN")
    p.add_argument("--csv", default="scripts/test_gnn.csv", help="Path to the prediction CSV")
    p.add_argument("--model_weights", default="weights/best_model.pth", help="Model weights (.pth)")
    p.add_argument("--out-predictions", default="scripts/FINAL_RESULTS.csv", help="Directory where the results will be saved.")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--pool", default="add", choices=["add", "mean"])
    args = p.parse_args()

    diretoria_base = Path(".").resolve()
    target_repo = diretoria_base / "repository"

    print("\n=== STEP 1: DATASET AND SUBDIRECTORIES ===")

    # Chama a função centralizada do utils_pipeline
    download_and_extract_all(target_repo)

    num_trees = len([f for f in target_repo.glob("*.nwk") if not f.name.startswith("._")])
    if num_trees == 0:
        sys.exit("\n[Erro] No valid tree was found in the folder 'repository'.")

    # Executa o filtro inteligente reconhecendo os caminhos
    csv_pronto = filtrar_csv_por_ficheiros_existentes(Path(args.csv), diretoria_base)

    print(f"\n=== STEP 2: GNN PREDICTION EXECUTION ({num_trees} mapped trees) ===")

    sys.argv = [
        "predict.py",
        "--csv", str(csv_pronto),
        "--zenodo_scripts", str(target_repo),
        "--model_weights", args.model_weights,
        "--out-predictions", args.out_predictions,
        "--hidden-dim", str(args.hidden_dim),
        "--pool", args.pool
    ]

    try:
        run_prediction()
    finally:
        if csv_pronto != Path(args.csv) and csv_pronto.exists():
            os.remove(csv_pronto)

    print("\n=== PROCESS COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()