import os
import sys
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from teacher import TeacherSSM

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "speech_commands"
MODEL_DIR = PROJECT_DIR / "models"

EXCLUDED_DIRS = {"_background_noise_"}
SAMPLE_RATE = 16000
MAX_LENGTH = 16000


class SpeechCommandsDataset(Dataset):
    def __init__(self, split="training"):
        self.samples = []
        self.labels = []
        self.class_names = sorted([
            d.name for d in DATASET_DIR.iterdir()
            if d.is_dir() and d.name not in EXCLUDED_DIRS
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        #Load official validation/test lists
        val_list = self._load_list("validation_list.txt")
        test_list = self._load_list("testing_list.txt")

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
                self.samples.append(wav_file)
                self.labels.append(self.class_to_idx[class_name])

    def _load_list(self, filename):
        list_path = DATASET_DIR / filename
        if not list_path.exists():
            return set()
        with open(list_path) as f:
            return set(line.strip() for line in f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, sr = sf.read(self.samples[idx], dtype="float32")
        #Pad or trim to MAX_LENGTH
        if len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
        else:
            audio = audio[:MAX_LENGTH]
        return torch.tensor(audio), self.labels[idx]


scaler = torch.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for audio, labels in tqdm(loader, desc="Training"):
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(audio)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * audio.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += audio.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, labels in tqdm(loader, desc="Evaluating"):
            audio = audio.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(audio)
                loss = criterion(logits, labels)

            total_loss += loss.item() * audio.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += audio.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #Data
    train_dataset = SpeechCommandsDataset("training")
    val_dataset = SpeechCommandsDataset("validation")
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Classes:", len(train_dataset.class_names))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    #Model
    model = TeacherSSM(n_classes=len(train_dataset.class_names)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print("Teacher parameters:", param_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    #Training
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        print()
        print("Epoch", epoch, "/", epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print("  Train loss:", round(train_loss, 4), " acc:", round(train_acc, 4))
        print("  Val   loss:", round(val_loss, 4), " acc:", round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODEL_DIR / "teacher_best.pt"
            torch.save(model.state_dict(), save_path)
            print("  Saved best model, val acc:", round(best_val_acc, 4))

    print()
    print("Training complete. Best val acc:", round(best_val_acc, 4))


if __name__ == "__main__":
    main()
    input("Press Enter to exit the programm")
