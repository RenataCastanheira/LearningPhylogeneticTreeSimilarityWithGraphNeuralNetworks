import sys
import zipfile
import shutil
import requests
import pandas as pd
from pathlib import Path


def extrair_e_normalizar_arvores(extracted_folder: Path, target_repo: Path):
    """
        Scans the extracted sub-zip files on disk, extracts all .nwk files, and
        saves them to the 'repository' folder after removing the _rr suffix from the name.
    """
    print("[Pipeline] Processing, extracting, and removing _rr suffixes from the trees...")

    for sub_zip in list(extracted_folder.rglob("*.zip")):
        if sub_zip.name == "phylogenetic_tree_dataset.zip":
            continue

        try:
            with zipfile.ZipFile(sub_zip, "r") as sz:
                sz.extractall(sub_zip.parent)
        except Exception as e:
            print(f"  [!] Warning extracting sub-zip {sub_zip.name}: {e}")

    contagem = 0
    for ficheiro_nwk in list(extracted_folder.rglob("*.nwk")):
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


def validate_and_filter_csv(csv_path: Path, base_dir: Path, tipo_csv: str):
    """
        Reads a specific CSV file, checks whether the referenced tree files exist on disk,
        and generates a temporary CSV file containing only the valid rows.
    """

    print(f"[Pipeline] Verifying consistency of {tipo_csv} CSV file '{csv_path.name}'...")

    if not csv_path.exists():
        sys.exit(f"[Error] Specified {tipo_csv} CSV file does not exist: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"[Erro] Failure reading {tipo_csv} CSV file: {e}")

    cols = [col for col in df.columns if col.lower().strip() in ['patha', 'pathb', 'tree_1', 'tree_2']]
    if len(cols) < 2:
        return csv_path

    col_a, col_b = cols[0], cols[1]
    valid_rows = []

    for idx, row in df.iterrows():
        p_a = str(row[col_a]).strip().replace("zenodo_scripts/", "", 1)
        p_b = str(row[col_b]).strip().replace("zenodo_scripts/", "", 1)

        if (base_dir / "repository" / p_a).exists() and (base_dir / "repository" / p_b).exists():
            valid_rows.append(row)

    if len(valid_rows) == 0:
        return csv_path

    df_filtrado = pd.DataFrame(valid_rows)
    csv_temporario = csv_path.parent / f"temp_{tipo_csv}_{csv_path.name}"
    df_filtrado.to_csv(csv_temporario, index=False)
    return csv_temporario