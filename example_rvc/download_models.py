"""
Python script to download the ai models
needed by linguflex from Huggingface's model hub.
"""

from huggingface_hub import hf_hub_download
import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_directories():
    create_directory("assets")
    create_directory("assets/hubert")
    create_directory("models")
    create_directory("models/rvc")
    create_directory("models/xtts")
    create_directory("models/xtts/Lasinya")


def download_file(url, filename_server, path_local):
    local_file = os.path.join(path_local, filename_server)
    if os.path.exists(local_file):
        print(f"File {filename_server} already exists in {path_local}.")
        return

    print(f"Downloading {filename_server} from repo {url} to {path_local}")
    hf_hub_download(repo_id=url, filename=filename_server, local_dir=path_local)


create_directories()

# download rvc base model (hubert) files
print("Downloading hubert base model files")
download_file("KoljaB/RVC_Assets", "hubert_base.pt", "assets/hubert")
download_file("KoljaB/RVC_Assets", "hubert_inputs.pth", "assets/hubert")

# download rvc trained model files
print("Downloading rvc trained model files")
download_file("KoljaB/RVC_Models", "Lasinya.pth", "models/rvc")
download_file("KoljaB/RVC_Models", "Lasinya.index", "models/rvc")

# download xtts trained model files
print("Downloading xtts trained model files (Lasinya)")
download_file("KoljaB/XTTS_Lasinya", "config.json", "models/xtts/Lasinya")
download_file("KoljaB/XTTS_Lasinya", "vocab.json", "models/xtts/Lasinya")
download_file("KoljaB/XTTS_Lasinya", "speakers_xtts.pth", "models/xtts/Lasinya")
download_file("KoljaB/XTTS_Lasinya", "model.pth", "models/xtts/Lasinya")
