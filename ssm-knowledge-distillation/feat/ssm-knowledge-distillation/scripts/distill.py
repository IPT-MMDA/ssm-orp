import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from itertools import product
from sklearn.metrics import f1_score
from tqdm import tqdm

from torch.utils.data import Dataset
from teacher import TeacherSSM
from student import StudentSSM
from dataset import SpeechCommandsDataset, MODEL_DIR


class KDDataset(Dataset):
    """Wraps audio + labels + cached teacher logits into one dataset."""
    def __init__(self, audio_dataset, teacher_logits):
        self.audios = audio_dataset.audios
        self.labels = audio_dataset.labels
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audios[idx], self.labels[idx], self.teacher_logits[idx]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CSV_PATH = PROJECT_DIR / "distillation_results.csv"

#grid search hyperparameters
TEMPERATURES = [1, 2, 5, 10]
ALPHAS = [0.0, 0.5, 1.0]


CSV_FIELDS = ["temperature", "alpha", "seed", "best_acc", "best_f1"]


def load_completed_runs():
    """Load already completed runs from CSV. Returns set of (temperature, alpha, seed) tuples."""
    completed = set()
    if not CSV_PATH.exists():
        return completed
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["temperature"]), float(row["alpha"]), int(row["seed"]))
            completed.add(key)
    return completed


def append_csv_row(temperature, alpha, seed, best_acc, best_f1):
    """Append a single run result to CSV, creating header if needed."""
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "temperature": temperature,
            "alpha": alpha,
            "seed": seed,
            "best_acc": round(best_acc, 4),
            "best_f1": round(best_f1, 4),
        })


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    #soft targets: KL divergence between softened distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    #hard targets: standard cross entropy
    ce = F.cross_entropy(student_logits, labels)

    return alpha * kl + (1 - alpha) * ce


scaler = torch.amp.GradScaler()


def train_one_epoch(student, loader, optimizer, temperature, alpha, device):
    student.train()
    total_loss = 0
    correct = 0
    total = 0

    for audio, labels, teacher_logits in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        teacher_logits = teacher_logits.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            student_logits = student(audio)
            loss = kd_loss(student_logits, teacher_logits, labels, temperature, alpha)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * audio.size(0)
        correct += (student_logits.argmax(dim=1) == labels).sum().item()
        total += audio.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(audio)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += audio.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return correct / total, macro_f1


def run_single_config(kd_train_loader, val_loader, n_classes, temperature, alpha, seed, device, epoch_bar):
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = StudentSSM(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=0.01)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_bar.reset(total=epochs)
    epoch_bar.set_description("T=" + str(temperature) + " a=" + str(alpha) + " s=" + str(seed))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            student, kd_train_loader, optimizer, temperature, alpha, device
        )
        scheduler.step()
        epoch_bar.set_postfix(loss=round(train_loss, 4), acc=round(train_acc, 4))
        epoch_bar.update(1)

    #evaluate once at the end
    val_acc, val_f1 = evaluate(student, val_loader, device)

    #save final model
    model_name = "student_T" + str(temperature) + "_a" + str(alpha) + "_s" + str(seed) + ".pt"
    torch.save(student.state_dict(), MODEL_DIR / model_name)

    return val_acc, val_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #data
    train_dataset = SpeechCommandsDataset("training")
    val_dataset = SpeechCommandsDataset("validation")
    n_classes = len(train_dataset.class_names)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Classes:", n_classes)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    #load frozen teacher
    teacher = TeacherSSM(n_classes=n_classes).to(device)
    teacher_path = MODEL_DIR / "teacher_best.pt"
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Loaded teacher from", teacher_path)

    teacher_acc, teacher_f1 = evaluate(teacher, val_loader, device)
    print("Teacher val acc:", round(teacher_acc, 4), " macro_f1:", round(teacher_f1, 4))

    #cache teacher logits in dataset order (unshuffled)
    print("Caching teacher logits ...")
    cache_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    all_teacher_logits = []
    with torch.no_grad():
        for audio, labels in tqdm(cache_loader, desc="Teacher logits"):
            audio = audio.to(device)
            with torch.amp.autocast("cuda"):
                logits = teacher(audio)
            all_teacher_logits.append(logits.float().cpu())
    cached_teacher_logits = torch.cat(all_teacher_logits, dim=0)
    print("Cached teacher logits:", cached_teacher_logits.shape)
    del teacher, cache_loader  #free GPU memory

    #build KD dataloader with cached logits
    kd_dataset = KDDataset(train_dataset, cached_teacher_logits)
    kd_train_loader = DataLoader(kd_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [42, 123]
    completed = load_completed_runs()

    all_runs = [(t, a, s) for t, a in product(TEMPERATURES, ALPHAS) for s in seeds]
    pending = [r for r in all_runs if r not in completed]

    if completed:
        print()
        print("Found", len(completed), "completed runs in CSV, skipping them")
    print()
    print("Pending runs:", len(pending), "/", len(all_runs))

    if not pending:
        print("All runs already complete.")
    else:
        run_bar = tqdm(total=len(pending), desc="Grid search", position=0)
        epoch_bar = tqdm(total=10, desc="Epochs", position=1, leave=False)

        for temp, alpha, seed in pending:
            best_acc, best_f1 = run_single_config(
                kd_train_loader, val_loader, n_classes, temp, alpha, seed, device, epoch_bar
            )
            append_csv_row(temp, alpha, seed, best_acc, best_f1)
            run_bar.update(1)
            run_bar.set_postfix(acc=round(best_acc, 4), f1=round(best_f1, 4))

        epoch_bar.close()
        run_bar.close()

    #print summary from CSV
    print()
    print("=== Summary ===")
    print("Teacher val acc:", round(teacher_acc, 4), " macro_f1:", round(teacher_f1, 4))
    print()
    print("Results saved to", CSV_PATH)
    print()
    print("T\talpha\tseed\tacc\tf1")
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row["temperature"], "\t", row["alpha"], "\t", row["seed"], "\t",
                  row["best_acc"], "\t", row["best_f1"])


if __name__ == "__main__":
    main()
    input("Press Enter to exit the programm")
