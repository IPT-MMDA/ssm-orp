"""Centralised configuration dataclasses.

Every hyperparameter lives here so experiment scripts never contain magic
numbers.  Dataclasses are JSON-serialisable, easy to copy between runs, and
give IDE type-checking for free.
"""

from dataclasses import dataclass, field


@dataclass
class PrefixConfig:
    """Settings for the trainable prefix vectors.

    Two parameterisation modes are supported:
    - Direct   (projection=False): a plain nn.Embedding of shape [K, D].
      Simple and fast; K*D trainable parameters.
    - Projected (projection=True): a smaller embedding [K, proj_hidden]
      passed through a 2-layer MLP → [K, D].  Li & Liang (2021) found this
      stabilises optimisation when D is large.
    """

    # Number of trainable prefix vectors prepended to each input sequence.
    prefix_length: int = 10

    # Must match the backbone's hidden size; set automatically in build_prefix_model.
    hidden_size: int = 768

    # Std of the normal distribution used to initialise prefix embeddings.
    init_std: float = 0.02

    # Use 2-layer MLP reparameterisation (Li & Liang 2021).
    projection: bool = False
    projection_hidden_size: int = 512


@dataclass
class PeriodicInjectionConfig:
    """Settings for the periodic prefix re-injection strategy.

    Background — the fading-memory problem:
        In a Mamba SSM the recurrence h_t = A·h_{t-1} + B·x_t means a prefix
        injected at position 0 contributes A^t · h_prefix to the hidden state
        at position t.  If A's spectral radius is < 1 (which Mamba encourages
        via its initialisation), this product decays exponentially, so the
        prefix is effectively "forgotten" after a few dozen tokens.

    Mitigation — periodic injection:
        Re-insert the same prefix block every `period` real-token positions.
        The maximum SSM-memory distance from any real token to the nearest
        prefix is then bounded by `period − 1` rather than the full sequence
        length T.  In other words, the worst-case attenuation is A^period
        instead of A^T.
    """

    enabled: bool = False

    # Insert a copy of the prefix every `period` real-token positions.
    # Smaller = stronger signal, but larger effective sequence length.
    period: int = 64


@dataclass
class LoraHyperparams:
    """LoRA configuration forwarded to peft.LoraConfig."""

    # Rank of the low-rank decomposition matrices.
    r: int = 8

    # Scaling factor α; effective learning rate scale = α / r.
    lora_alpha: int = 16

    lora_dropout: float = 0.1

    # Linear projection layers inside each MambaMixer block.
    # in_proj / out_proj handle the gated MLP path;
    # x_proj produces B, C, Δ for the SSM kernel.
    target_modules: list[str] = field(
        default_factory=lambda: ["in_proj", "out_proj", "x_proj"]
    )


@dataclass
class TrainingConfig:
    """Top-level training settings shared across all three methods."""

    # One of "prefix", "prefix_periodic", "lora", "full".
    method: str = "prefix"

    # HuggingFace model name.  mamba-130m-hf is the smallest Mamba checkpoint
    # that ships with a sequence-classification head config.
    model_name: str = "state-spaces/mamba-130m-hf"

    num_epochs: int = 5
    batch_size: int = 16

    # Prefix tuning benefits from a higher LR than LoRA / full fine-tuning
    # because prefix parameters start random and must adapt quickly.
    learning_rate: float = 3e-4

    weight_decay: float = 0.01

    # Sequences longer than this are truncated; kept small to speed up runs.
    max_seq_len: int = 128

    seed: int = 42
    output_dir: str = "results/"

    # Bootstrap confidence interval settings (used in evaluator.py).
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95

    # Number of DataLoader worker processes.  0 = main process (safe on all OS).
    num_workers: int = 0

    # Compute device.  "auto" = use CUDA if available, else CPU.
    # Pass "cpu", "cuda", or "cuda:0" etc. to force a specific device.
    device: str = "auto"


@dataclass
class EpochResult:
    """Metrics logged at the end of each training epoch.

    Stored here (rather than in trainer.py) so it can be imported without
    pulling in torch — useful for analysis scripts and unit tests.
    """

    epoch: int
    train_loss: float
    val_accuracy: float
    val_f1: float
    # 95 % bootstrap confidence interval bounds on val_accuracy.
    ci_lower: float
    ci_upper: float
