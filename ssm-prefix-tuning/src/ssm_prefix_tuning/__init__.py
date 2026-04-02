"""ssm_prefix_tuning — Prefix Tuning vs LoRA vs Full Fine-Tuning on Mamba SSMs.

Heavy dependencies (torch, transformers, peft) are imported lazily so that
lightweight submodules (config, evaluator, utils) can be used without a full
ML stack installed — e.g. in unit tests or data-analysis notebooks.
"""

# Config and pure-Python utilities are always available.
from ssm_prefix_tuning.config import (  # noqa: F401
    EpochResult,
    LoraHyperparams,
    PeriodicInjectionConfig,
    PrefixConfig,
    TrainingConfig,
)
from ssm_prefix_tuning.evaluator import (  # noqa: F401
    bootstrap_ci,
    compare_methods,
    plot_prefix_length_vs_accuracy,
    prefix_length_sweep_table,
)
from ssm_prefix_tuning.utils import (  # noqa: F401
    count_parameters,
    get_device,
    save_results,
    set_seed,
)

# Torch-dependent submodules are imported on demand to avoid hard failures when
# torch / transformers / peft are not installed (e.g. during CI lint steps).
def _lazy_import(name: str):
    import importlib
    return importlib.import_module(f"ssm_prefix_tuning.{name}")


def __getattr__(attr: str):
    # Map each public name to its submodule.
    _attr_module = {
        "PrefixEncoder": "prefix_encoder",
        "PeriodicPrefixInjector": "prefix_encoder",
        "MambaPrefixModel": "model_wrapper",
        "build_prefix_model": "model_wrapper",
        "build_prefix_model_from_config": "model_wrapper",
        "build_full_finetune_model": "model_wrapper",
        "build_lora_model": "lora_model",
        "count_trainable_parameters": "lora_model",
        "build_optimizer": "trainer",
        "run_training": "trainer",
        "train_one_epoch": "trainer",
        "evaluate": "trainer",
        "load_sst2": "data",
        "get_tokenizer": "data",
    }
    if attr in _attr_module:
        mod = _lazy_import(_attr_module[attr])
        return getattr(mod, attr)
    raise AttributeError(f"module 'ssm_prefix_tuning' has no attribute {attr!r}")


__all__ = [
    # Config
    "PrefixConfig", "PeriodicInjectionConfig", "LoraHyperparams", "TrainingConfig",
    # Prefix encoder
    "PrefixEncoder", "PeriodicPrefixInjector",
    # Models
    "MambaPrefixModel", "build_prefix_model", "build_prefix_model_from_config",
    "build_full_finetune_model", "build_lora_model", "count_trainable_parameters",
    # Training
    "EpochResult", "build_optimizer", "run_training", "train_one_epoch", "evaluate",
    # Evaluation
    "bootstrap_ci", "compare_methods", "prefix_length_sweep_table",
    "plot_prefix_length_vs_accuracy",
    # Data
    "load_sst2", "get_tokenizer",
    # Utils
    "set_seed", "get_device", "count_parameters", "save_results",
]
