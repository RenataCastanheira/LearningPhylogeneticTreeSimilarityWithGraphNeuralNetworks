import os
import io
import sys
import zipfile
import shutil
import requests
from pathlib import Path


def download_zenodo(record_id, extract_to="/app/input_data"):
    url = f"https://zenodo.org/api/records/{record_id}"
    print(f"[Zenodo] Searching registry {record_id} in Zenodo...")

    try:
        response = requests.get(url).json()
    except Exception as e:
        print(f"[Zenodo] Error connecting to the API: {e}")
        sys.exit(1)

    # Procura pelo ficheiro ZIP principal no registo
    zipfile_info = None
    for file_info in response.get('files', []):
        if file_info['key'].endswith(".zip") or 'dataset' in file_info['key'].lower():
            zipfile_info = file_info
            break

    if not zipfile_info:
        print(f"[Zenodo] Error: No valid ZIP archive found in registry {record_id}.")
        sys.exit(1)

    zip_filename = zipfile_info['key']
    download_path = zipfile_info['links']['self']

    base_path = Path(extract_to)
    target_repo = base_path / "repository"

    # Se a pasta repository já existir e contiver ficheiros .nwk, salta o processo
    if target_repo.exists() and any(target_repo.glob("*.nwk")):
        print(f"[Zenodo] The directory '{target_repo}' already contains trees .nwk. Download ignored.")
        return

    # Garante que a pasta repository final existe localmente
    target_repo.mkdir(parents=True, exist_ok=True)

    print(f"[Zenodo] Downloading {zip_filename}...")
    r = requests.get(download_path, stream=True)
    if r.status_code != 200:
        print(f"[Zenodo] Error in download: Status {r.status_code}")
        sys.exit(1)

    # Carrega o ZIP principal diretamente na memória para processamento rápido
    print("[Zenodo] Download concluded. Processing principal ZIP...")
    zip_bytes = io.BytesIO()
    for chunk in r.iter_content(chunk_size=8192):
        zip_bytes.write(chunk)
    zip_bytes.seek(0)

    with zipfile.ZipFile(zip_bytes, "r") as zip_principal:
        # Percorre a estrutura interna do ZIP principal à procura dos sub-zips
        for caminho_interno in zip_principal.namelist():
            if caminho_interno.endswith(".zip"):
                print(f"[Auto-Extractor] Opening internal sub-zip: {caminho_interno}")

                # Lê o sub-zip para a memória
                sub_zip_data = zip_principal.read(caminho_interno)
                try:
                    with zipfile.ZipFile(io.BytesIO(sub_zip_data)) as sub_zip:
                        # Extrai individualmente os ficheiros .nwk de cada sub-zip
                        for ficheiro_sub in sub_zip.infolist():
                            if ficheiro_sub.filename.endswith(".nwk"):
                                # Limpa subpastas internas e obtém apenas o nome do ficheiro .nwk
                                nome_ficheiro = Path(ficheiro_sub.filename).name
                                caminho_final = target_repo / nome_ficheiro

                                # Copia o ficheiro .nwk diretamente para a pasta repository
                                with sub_zip.open(ficheiro_sub) as fonte, open(caminho_final, "wb") as destino:
                                    shutil.copyfileobj(fonte, destino)
                except Exception as e:
                    print(f"[Auto-Extractor] Error extracting sub-zip {caminho_interno}: {e}")

    print(f"[Zenodo] Success! All the phylogenetic trees (.nwk) were organized in: {target_repo}")


if __name__ == "__main__":
    ZENODO_ID = "20476872"
    download_zenodo(ZENODO_ID, extract_to="/app/input_data")