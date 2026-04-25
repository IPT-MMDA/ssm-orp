import json
import os
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACT_DIR = "artifacts"


def main():
    csv_path = os.path.join(ARTIFACT_DIR, "results.csv")
    json_path = os.path.join(ARTIFACT_DIR, "results.json")

    df = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "baseline_accuracy" in df.columns:
        # clean vs scenarios: accuracy
        plt.figure(figsize=(10, 5))
        plt.bar(df["scenario"], df["baseline_accuracy"], label="Baseline", alpha=0.8)
        plt.bar(df["scenario"], df["quant_accuracy"], label="INT8", alpha=0.8)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Baseline vs INT8 across scenarios")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "accuracy_comparison.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(df["scenario"], df["baseline_f1"], label="Baseline", alpha=0.8)
        plt.bar(df["scenario"], df["quant_f1"], label="INT8", alpha=0.8)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("F1")
        plt.title("Baseline vs INT8 F1 across scenarios")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "f1_comparison.png"))
        plt.close()
    else:
        plt.figure(figsize=(10, 5))
        plt.bar(df["scenario"], df["baseline_mse"], label="Baseline", alpha=0.8)
        plt.bar(df["scenario"], df["quant_mse"], label="INT8", alpha=0.8)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("MSE")
        plt.title("Baseline vs INT8 MSE across scenarios")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "mse_comparison.png"))
        plt.close()

    # memory + latency
    memory = data["memory"]
    latency = data["latency"]

    plt.figure(figsize=(6, 4))
    plt.bar(["FP32 A/B/C", "INT8 A/B/C"], [memory["abc_fp32_bytes"], memory["abc_int8_bytes"]])
    plt.ylabel("Bytes")
    plt.title("Memory of A, B, C")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "memory_abc.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Baseline", "INT8"], [latency["baseline_avg_seconds"], latency["quant_avg_seconds"]])
    plt.ylabel("Seconds")
    plt.title("Average inference latency")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "latency.png"))
    plt.close()

    print("Saved plots to artifacts/")


if __name__ == "__main__":
    main()
