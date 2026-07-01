import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import argparse
import zipfile
import shutil
import requests
import pandas as pd
from pathlib import Path

diretoria_tool = os.path.dirname(os.path.abspath(__file__))
diretoria_pai_gnn = os.path.dirname(diretoria_tool)

if diretoria_tool not in sys.path:
    sys.path.insert(0, diretoria_tool)
if diretoria_pai_gnn not in sys.path:
    sys.path.insert(0, diretoria_pai_gnn)

# Agora o import vai funcionar perfeitamente!
from predict import main as run_prediction

def filtrar_csv_por_ficheiros_existentes(caminho_csv: Path, base_dir: Path):
    """
    Reads the input CSV, verifies whether the referenced tree files exist on disk, and
    generates a temporary CSV containing only the rows whose files exist in the repository.
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


def extrair_e_normalizar_arvores(pasta_extraida: Path, target_repo: Path):
    """
    Varre os sub-zips extraídos no disco, retira todos os ficheiros .nwk e
    salva-os na pasta 'repository' removendo o sufixo _rr do nome.
    """
    print("[Pipeline] Processing, extracting, and removing _rr suffixes from the trees...")

    for sub_zip in list(pasta_extraida.rglob("*.zip")):
        if sub_zip.name == "phylogenetic_tree_dataset.zip":
            continue

        try:
            with zipfile.ZipFile(sub_zip, "r") as sz:
                sz.extractall(sub_zip.parent)
        except Exception as e:
            print(f"  [!] Warning extracting sub-zip {sub_zip.name}: {e}")

    contagem = 0
    for ficheiro_nwk in list(pasta_extraida.rglob("*.nwk")):
        nome_original = ficheiro_nwk.name

        if nome_original.startswith("._") or "__MACOSX" in str(ficheiro_nwk):
            continue

        if "_rr.nwk" in nome_original:
            nome_limpo = nome_original.replace("_rr.nwk", ".nwk")
        else:
            nome_limpo = nome_original

        caminho_final = target_repo / nome_limpo
        shutil.copy2(str(ficheiro_nwk), str(caminho_final))
        contagem += 1

    return contagem


def download_and_extract_all(target_repo: Path):
    """
    Ensures that the correct zip file is downloaded and
    organizes the trees cleanly.
    """
    # Correção crucial: Se a pasta física não existir ou se não tiver nenhum .nwk legítimo lá dentro, recria tudo do zero!
    if target_repo.exists() and any(target_repo.glob("*.nwk")):
        print(f"[Dataset] The folder'{target_repo.name}' already contains the prepared trees. Proceeding...")
        return

    print(f"[Dataset] Directory '{target_repo.name}' not found or empty. Creating repository...")
    if target_repo.exists():
        shutil.rmtree(target_repo)
    target_repo.mkdir(parents=True, exist_ok=True)

    zip_alvo_path = Path("phylogenetic_tree_dataset.zip")
    pasta_temp_extracao = Path("temp_dataset_extracted")

    if zip_alvo_path.exists():
        print(f"[Dataset] Founded archive '{zip_alvo_path.name}' already on disk. Processing...")
    else:
        print("[Dataset] Starting automatic download from Zenodo...")
        record_id = "20476872"
        url = f"https://zenodo.org/api/records/{record_id}"

        try:
            response = requests.get(url).json()
            file_info = None
            for f in response.get('files', []):
                if f['key'] == "phylogenetic_tree_dataset.zip":
                    file_info = f
                    break

            if not file_info:
                sys.exit("[Dataset] Error: File 'phylogenetic_tree_dataset.zip' not found on Zenodo.")

            download_url = file_info['links']['self']
            tamanho_gb = file_info.get('size', 0) / (1024 * 1024 * 1024)

            print(f"[Dataset] Target confirmed: {file_info['key']} (~{tamanho_gb:.2f} GB)")
            print(f"[Dataset] Downloading the package to disk...")

            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(zip_alvo_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print("[Dataset] Download concluded with success!")
        except Exception as e:
            sys.exit(f"[Dataset] Critical error downloading from Zenodo: {e}")

    try:
        print("[Dataset] Extracting the main package into a temporary directory...")
        if pasta_temp_extracao.exists():
            shutil.rmtree(pasta_temp_extracao)

        with zipfile.ZipFile(zip_alvo_path, "r") as zip_ref:
            zip_ref.extractall(pasta_temp_extracao)

        total = extrair_e_normalizar_arvores(pasta_temp_extracao, target_repo)

        print("[Dataset] Cleaning up temporary directories...")
        if pasta_temp_extracao.exists():
            shutil.rmtree(pasta_temp_extracao)

        print(f"[Dataset] Success! {total} trees organized and normalized in {target_repo.name}")

    except Exception as e:
        if pasta_temp_extracao.exists():
            shutil.rmtree(pasta_temp_extracao)
        sys.exit(f"[Dataset] Critical error during extraction: {e}")


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

    # Executa a rotina inteligente que valida se a pasta existe de facto física no Mac
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
        "--zenodo_scripts", str(diretoria_base),
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