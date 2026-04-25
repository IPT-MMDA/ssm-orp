from dataclasses import dataclass


@dataclass
class DataConfig:
    task_type: str = "classification"   # "classification" or "regression"
    num_train: int = 3000
    num_val: int = 800
    num_test: int = 800
    seq_len: int = 40
    input_dim: int = 6
    hidden_dim: int = 16
    ssm_output_dim: int = 8
    num_classes: int = 3
    batch_size: int = 64
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    save_dir: str = "artifacts"
    checkpoint_name: str = "baseline_best.pt"


@dataclass
class QuantConfig:
    num_bits: int = 8
    symmetric: bool = True
    calibration_batches: int = 10
    gaussian_noise_sigmas: tuple = (0.05, 0.10)
    shift_values: tuple = (1, 3)
    benchmark_warmup: int = 20
    benchmark_runs: int = 100
