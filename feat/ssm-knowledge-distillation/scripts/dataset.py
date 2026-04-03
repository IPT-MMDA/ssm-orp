import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "speech_commands"
CACHE_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"

EXCLUDED_DIRS = {"_background_noise_"}
SAMPLE_RATE = 16000
MAX_LENGTH = 16000


class SpeechCommandsDataset(Dataset):
    def __init__(self, split="training"):
        self.class_names = sorted([
            d.name for d in DATASET_DIR.iterdir()
            if d.is_dir() and d.name not in EXCLUDED_DIRS
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        cache_path = CACHE_DIR / (split + ".pt")
        if cache_path.exists():
            print("Loading", split, "from cache ...")
            cache = torch.load(cache_path, weights_only=True)
            self.audios = cache["audios"]
            self.labels = cache["labels"]
            return

        val_list = self._load_list("validation_list.txt")
        test_list = self._load_list("testing_list.txt")

        #collect file paths first
        paths = []
        labels = []
        for class_name in self.class_names:
            class_dir = DATASET_DIR / class_name
            for wav_file in class_dir.glob("*.wav"):
                rel_path = class_name + "/" + wav_file.name
                if split == "validation" and rel_path not in val_list:
                    continue
                if split == "testing" and rel_path not in test_list:
                    continue
                if split == "training" and (rel_path in val_list or rel_path in test_list):
                    continue
                paths.append(wav_file)
                labels.append(self.class_to_idx[class_name])

        #preload all audio into RAM
        print("Loading", split, "split from wav files ...")
        self.audios = torch.zeros(len(paths), MAX_LENGTH)
        self.labels = torch.tensor(labels, dtype=torch.long)
        for i, path in enumerate(tqdm(paths, desc=split)):
            audio, sr = sf.read(path, dtype="float32")
            length = min(len(audio), MAX_LENGTH)
            self.audios[i, :length] = torch.from_numpy(audio[:length])

        #save cache for next time
        print("Saving", split, "cache to", cache_path)
        torch.save({"audios": self.audios, "labels": self.labels}, cache_path)

    def _load_list(self, filename):
        list_path = DATASET_DIR / filename
        if not list_path.exists():
            return set()
        with open(list_path) as f:
            return set(line.strip() for line in f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audios[idx], self.labels[idx]
