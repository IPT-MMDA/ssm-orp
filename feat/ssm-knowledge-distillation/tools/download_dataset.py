"""
Download Speech Commands v2 dataset (35 classes).
Downloads to feat/ssm-knowledge-distillation/dataset/speech_commands/
"""

import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "speech_commands"
ARCHIVE_PATH = PROJECT_DIR / "dataset" / "speech_commands_v0.02.tar.gz"


def _ask_yes_no(prompt):
    yes = {"y", "yes"}
    no = {"n", "no", "nope"}
    while True:
        answer = input(prompt + " (Y/N): ").strip().lower()
        if answer in yes:
            return True
        if answer in no:
            return False
        print("Please enter Y or N.")


def download_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print("Dataset already exists at", DATASET_DIR, "skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: show download size, ask to proceed
    print("Fetching download info ...")
    req = urllib.request.urlopen(DATASET_URL)
    total_size = int(req.headers.get("Content-Length", 0))
    download_mb = round(total_size / 1024 / 1024)
    print("Download size:", download_mb, "MB")

    if not _ask_yes_no("Proceed with download?"):
        req.close()
        return

    # Step 2: download
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
    with open(ARCHIVE_PATH, "wb") as f:
        while True:
            chunk = req.read(8192)
            if not chunk:
                break
            f.write(chunk)
            progress.update(len(chunk))
    progress.close()
    req.close()

    # Step 3: show unpacked size, ask to proceed
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        members = tar.getmembers()
    unpacked_bytes = sum(m.size for m in members)
    unpacked_mb = round(unpacked_bytes / 1024 / 1024)
    print("Unpacked size:", unpacked_mb, "MB")

    if not _ask_yes_no("Proceed with extraction?"):
        ARCHIVE_PATH.unlink()
        print("Archive removed, extraction skipped.")
        return

    # Step 4: extract
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=DATASET_DIR)

    ARCHIVE_PATH.unlink()
    print("Removed archive file.")
    print("Done!")
    input("Press Enter to exit the programm")


if __name__ == "__main__":
    download_dataset()
