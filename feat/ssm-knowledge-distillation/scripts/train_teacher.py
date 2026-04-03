import os
import sys
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from teacher import TeacherSSM
from dataset import SpeechCommandsDataset, MODEL_DIR


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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels in tqdm(loader, desc="Evaluating"):
            audio = audio.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(audio)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * audio.size(0)
            correct += (preds == labels).sum().item()
            total += audio.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / total, correct / total, macro_f1


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
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print("  Train loss:", round(train_loss, 4), " acc:", round(train_acc, 4))
        print("  Val   loss:", round(val_loss, 4), " acc:", round(val_acc, 4), " macro_f1:", round(val_f1, 4))

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
