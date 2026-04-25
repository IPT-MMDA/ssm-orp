import os
import json
import pandas as pd
import torch

from config import DataConfig, TrainConfig, QuantConfig
from data import build_dataloaders
from model import build_model
from utils import set_seed, get_device, ensure_dir
from quant_utils import (
    get_calibration_batches,
    run_calibration,
    build_quantized_model,
    relative_weight_error,
)
from eval_utils import evaluate_model, robustness_degradation
from robustness_utils import add_gaussian_noise, shift_sequence
from benchmark_utils import (
    abc_fp32_memory_bytes,
    abc_int8_memory_bytes,
    memory_savings_percent,
    benchmark_latency,
    speedup_ratio,
)


def main():
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    quant_cfg = QuantConfig()

    set_seed(data_cfg.seed)
    device = get_device(train_cfg.device)
    ensure_dir(train_cfg.save_dir)

    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)

    model = build_model(data_cfg).to(device)
    checkpoint_path = os.path.join(train_cfg.save_dir, train_cfg.checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calibration_batches = get_calibration_batches(train_loader, num_batches=quant_cfg.calibration_batches)
    run_calibration(model, calibration_batches, device)

    q_model, quant_artifacts = build_quantized_model(
        model,
        symmetric=quant_cfg.symmetric,
        num_bits=quant_cfg.num_bits
    )
    q_model = q_model.to(device)
    q_model.eval()

    # Weight reconstruction errors
    weight_errors = {
        "A_rel_error": relative_weight_error(model.A.data, q_model.A.data),
        "B_rel_error": relative_weight_error(model.B.data, q_model.B.data),
        "C_rel_error": relative_weight_error(model.C.data, q_model.C.data),
    }

    # Clean evaluation
    baseline_clean = evaluate_model(model, test_loader, device, data_cfg.task_type)
    quant_clean = evaluate_model(q_model, test_loader, device, data_cfg.task_type)

    results_rows = []

    def add_row(name, baseline_metrics, quant_metrics):
        row = {"scenario": name}
        row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        row.update({f"quant_{k}": v for k, v in quant_metrics.items()})

        if data_cfg.task_type == "classification":
            row["delta_accuracy"] = quant_metrics["accuracy"] - baseline_metrics["accuracy"]
            row["delta_f1"] = quant_metrics["f1"] - baseline_metrics["f1"]
        else:
            row["delta_mse"] = quant_metrics["mse"] - baseline_metrics["mse"]
        results_rows.append(row)

    add_row("clean", baseline_clean, quant_clean)

    # Noise scenarios
    for sigma in quant_cfg.gaussian_noise_sigmas:
        baseline_noise = evaluate_model(
            model, test_loader, device, data_cfg.task_type,
            perturb_fn=add_gaussian_noise, perturb_kwargs={"sigma": sigma}
        )
        quant_noise = evaluate_model(
            q_model, test_loader, device, data_cfg.task_type,
            perturb_fn=add_gaussian_noise, perturb_kwargs={"sigma": sigma}
        )
        add_row(f"gaussian_noise_sigma_{sigma}", baseline_noise, quant_noise)

    # Shift scenarios
    for shift in quant_cfg.shift_values:
        baseline_shift = evaluate_model(
            model, test_loader, device, data_cfg.task_type,
            perturb_fn=shift_sequence, perturb_kwargs={"shift": shift}
        )
        quant_shift = evaluate_model(
            q_model, test_loader, device, data_cfg.task_type,
            perturb_fn=shift_sequence, perturb_kwargs={"shift": shift}
        )
        add_row(f"sequence_shift_{shift}", baseline_shift, quant_shift)

    # Robustness degradation relative to clean
    robustness_summary = {}
    if data_cfg.task_type == "classification":
        for row in results_rows:
            if row["scenario"] == "clean":
                continue
            base_metrics = {
                "accuracy": row["baseline_accuracy"],
                "f1": row["baseline_f1"]
            }
            quant_metrics = {
                "accuracy": row["quant_accuracy"],
                "f1": row["quant_f1"]
            }

            robustness_summary[row["scenario"]] = {
                "baseline_drop_vs_clean": robustness_degradation(
                    baseline_clean, base_metrics, task_type=data_cfg.task_type
                ),
                "quant_drop_vs_clean": robustness_degradation(
                    quant_clean, quant_metrics, task_type=data_cfg.task_type
                )
            }
    else:
        for row in results_rows:
            if row["scenario"] == "clean":
                continue
            base_metrics = {"mse": row["baseline_mse"]}
            quant_metrics = {"mse": row["quant_mse"]}
            robustness_summary[row["scenario"]] = {
                "baseline_drop_vs_clean": robustness_degradation(
                    baseline_clean, base_metrics, task_type=data_cfg.task_type
                ),
                "quant_drop_vs_clean": robustness_degradation(
                    quant_clean, quant_metrics, task_type=data_cfg.task_type
                )
            }

    # Memory
    fp_bytes = abc_fp32_memory_bytes(model)
    q_bytes = abc_int8_memory_bytes(quant_artifacts)
    mem_saving = memory_savings_percent(fp_bytes, q_bytes)

    # Latency
    sample_batch = next(iter(test_loader))
    baseline_latency = benchmark_latency(
        model, sample_batch, device,
        n_warmup=quant_cfg.benchmark_warmup,
        n_runs=quant_cfg.benchmark_runs
    )
    quant_latency = benchmark_latency(
        q_model, sample_batch, device,
        n_warmup=quant_cfg.benchmark_warmup,
        n_runs=quant_cfg.benchmark_runs
    )
    speedup = speedup_ratio(baseline_latency, quant_latency)

    results = {
        "task_type": data_cfg.task_type,
        "quantization": {
            "num_bits": quant_cfg.num_bits,
            "symmetric": quant_cfg.symmetric,
            "calibration_batches": quant_cfg.calibration_batches
        },
        "weight_errors": weight_errors,
        "memory": {
            "abc_fp32_bytes": fp_bytes,
            "abc_int8_bytes": q_bytes,
            "abc_memory_saving_percent": mem_saving
        },
        "latency": {
            "baseline_avg_seconds": baseline_latency,
            "quant_avg_seconds": quant_latency,
            "speedup_ratio": speedup
        },
        "scenario_results": results_rows,
        "robustness_summary": robustness_summary
        #"quant_artifacts": quant_artifacts,
    }

    json_path = os.path.join(train_cfg.save_dir, "results.json")
    csv_path = os.path.join(train_cfg.save_dir, "results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results_rows).to_csv(csv_path, index=False)

    print("\nSaved:")
    print(json_path)
    print(csv_path)
    print("\nMemory:", results["memory"])
    print("Latency:", results["latency"])
    print("Weight errors:", weight_errors)


if __name__ == "__main__":
    main()
