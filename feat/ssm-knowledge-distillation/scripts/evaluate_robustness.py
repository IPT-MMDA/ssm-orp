import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from student import StudentSSM
from dataset import SpeechCommandsDataset, MODEL_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CSV_PATH = PROJECT_DIR / "robustness_results.csv"
DISTILL_CSV = PROJECT_DIR / "distillation_results.csv"

CSV_FIELDS = [
    "temperature", "alpha", "seed",
    "clean_acc", "clean_f1",
    "fgsm_0.01_acc", "fgsm_0.01_f1",
    "fgsm_0.05_acc", "fgsm_0.05_f1",
    "fgsm_0.1_acc", "fgsm_0.1_f1",
    "noise_20db_acc", "noise_20db_f1",
    "noise_10db_acc", "noise_10db_f1",
    "noise_5db_acc", "noise_5db_f1",
    "trunc_25_acc", "trunc_25_f1",
    "trunc_50_acc", "trunc_50_f1",
    "trunc_75_acc", "trunc_75_f1",
]


def compute_metrics(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    return round(float(acc), 4), round(float(f1), 4)


def evaluate_clean(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(audio)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)
    return compute_metrics(torch.cat(all_preds), torch.cat(all_labels))


def evaluate_fgsm(model, loader, device, epsilon):
    """FGSM adversarial attack: perturb input to maximize loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []

    for audio, labels in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        audio.requires_grad = True

        with torch.amp.autocast("cuda"):
            logits = model(audio)
            loss = criterion(logits, labels)

        loss.backward()

        #fgsm perturbation
        perturbed = audio + epsilon * audio.grad.sign()
        perturbed = perturbed.detach()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = model(perturbed)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    return compute_metrics(torch.cat(all_preds), torch.cat(all_labels))


def evaluate_noise(model, loader, device, snr_db):
    """Add Gaussian noise at a given SNR level."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            #compute noise level from SNR
            signal_power = audio.pow(2).mean(dim=1, keepdim=True)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(audio) * noise_power.sqrt()
            noisy = audio + noise

            with torch.amp.autocast("cuda"):
                logits = model(noisy)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)

    return compute_metrics(torch.cat(all_preds), torch.cat(all_labels))


def evaluate_truncation(model, loader, device, trunc_pct):
    """Zero out the last trunc_pct% of the audio."""
    model.eval()
    all_preds = []
    all_labels = []
    seq_len = 16000

    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            #zero out the tail
            cut_point = int(seq_len * (1 - trunc_pct / 100))
            truncated = audio.clone()
            truncated[:, cut_point:] = 0

            with torch.amp.autocast("cuda"):
                logits = model(truncated)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)

    return compute_metrics(torch.cat(all_preds), torch.cat(all_labels))


def load_completed():
    completed = set()
    if not CSV_PATH.exists():
        return completed
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["temperature"]), float(row["alpha"]), int(row["seed"]))
            completed.add(key)
    return completed


def append_row(row_dict):
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def get_student_configs():
    """Read distillation CSV to get list of (temp, alpha, seed) configs."""
    configs = []
    with open(DISTILL_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append((int(row["temperature"]), float(row["alpha"]), int(row["seed"])))
    return configs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #data
    test_dataset = SpeechCommandsDataset("testing")
    n_classes = len(test_dataset.class_names)
    print("Test samples:", len(test_dataset))
    print("Classes:", n_classes)

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    #get configs and skip completed
    configs = get_student_configs()
    completed = load_completed()
    pending = [c for c in configs if c not in completed]

    if completed:
        print("Found", len(completed), "completed evaluations, skipping them")
    print("Pending:", len(pending), "/", len(configs))

    if not pending:
        print("All evaluations complete.")
    else:
        for temp, alpha, seed in tqdm(pending, desc="Evaluating models"):
            model_name = "student_T" + str(temp) + "_a" + str(alpha) + "_s" + str(seed) + ".pt"
            model_path = MODEL_DIR / model_name

            if not model_path.exists():
                print("Missing:", model_name, "- skipping")
                continue

            student = StudentSSM(n_classes=n_classes).to(device)
            student.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            student.eval()

            row = {
                "temperature": temp,
                "alpha": alpha,
                "seed": seed,
            }

            #clean
            acc, f1 = evaluate_clean(student, test_loader, device)
            row["clean_acc"] = acc
            row["clean_f1"] = f1

            #fgsm at different epsilons
            for eps in [0.01, 0.05, 0.1]:
                acc, f1 = evaluate_fgsm(student, test_loader, device, eps)
                row["fgsm_" + str(eps) + "_acc"] = acc
                row["fgsm_" + str(eps) + "_f1"] = f1

            #noise at different SNR levels
            for snr in [20, 10, 5]:
                acc, f1 = evaluate_noise(student, test_loader, device, snr)
                row["noise_" + str(snr) + "db_acc"] = acc
                row["noise_" + str(snr) + "db_f1"] = f1

            #truncation at different percentages
            for pct in [25, 50, 75]:
                acc, f1 = evaluate_truncation(student, test_loader, device, pct)
                row["trunc_" + str(pct) + "_acc"] = acc
                row["trunc_" + str(pct) + "_f1"] = f1

            append_row(row)
            del student

    #print summary
    print()
    print("=== Robustness Results ===")
    if CSV_PATH.exists():
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                print("T=", row["temperature"], "a=", row["alpha"], "s=", row["seed"],
                      "| clean:", row["clean_acc"],
                      "| fgsm0.05:", row["fgsm_0.05_acc"],
                      "| noise10db:", row["noise_10db_acc"],
                      "| trunc50:", row["trunc_50_acc"])


if __name__ == "__main__":
    main()
    input("Press Enter to exit the programm")
