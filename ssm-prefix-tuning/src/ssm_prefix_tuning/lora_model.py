"""LoRA fine-tuning factory for Mamba sequence classification.

LoRA (Hu et al., 2021) adds low-rank perturbation matrices ΔW = B·A to
selected linear layers.  Only B and A are trained; the pre-trained W is
frozen.  This gives a compact approximation to full fine-tuning.

Why LoRA works better on Mamba projections than on SSM modules:
    The MambaMixer block contains three types of linear layers:
        in_proj   — projects input x to (z, x') for the gated MLP path.
        out_proj  — projects the SSM output back to the residual stream.
        x_proj    — produces the B, C, Δ parameters for the SSM kernel.
    These are standard linear layers and are compatible with LoRA.
    The SSM recurrence itself (A, B, C, Δ) is implemented as parameter
    tensors with specialised structured updates — applying LoRA there
    is non-trivial and generally not beneficial (MambaPEFT, 2024).

Parameter efficiency comparison (approximate, mamba-130m-hf):
    Full fine-tuning : ~130 M parameters
    LoRA r=8         :   ~1–2 M parameters   (< 2 % of total)
    Prefix K=10      :    ~8 K parameters
"""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel

from ssm_prefix_tuning.config import LoraHyperparams
from ssm_prefix_tuning.model_wrapper import MambaClassifier


def build_lora_model(
    model_name: str,
    lora_cfg: LoraHyperparams,
    num_labels: int = 2,
) -> PeftModel:
    """Load a pre-trained Mamba model and apply LoRA adapters.

    Uses MambaClassifier (MambaModel + linear head) as the base model because
    MambaForSequenceClassification is not available in transformers >= 5.x.

    LoRA is applied to the linear projection layers inside each MambaMixer:
        in_proj  — input projection for the gated MLP path.
        out_proj — output projection back to the residual stream.
        x_proj   — produces B, C, Δ for the SSM kernel.

    The classification head (MambaClassifier.classifier) is kept fully
    trainable via modules_to_save so it adapts to the new task.

    Args:
        model_name: HuggingFace model identifier.
        lora_cfg:   LoraHyperparams — rank, alpha, dropout, target modules.
        num_labels: Number of output classes.

    Returns:
        PeftModel wrapping MambaClassifier.  Exposes the same
        forward(input_ids, attention_mask, labels) interface so the
        shared training loop in trainer.py works without modification.
    """
    base_model = MambaClassifier.from_pretrained(model_name, num_labels)

    peft_config = LoraConfig(
        # FEATURE_EXTRACTION avoids task-specific PEFT wrappers (e.g.
        # PeftModelForCausalLM) that require prepare_inputs_for_generation —
        # a method our custom MambaClassifier does not implement.
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        # modules_to_save: the classification head is saved and trained
        # alongside the LoRA adapters.
        modules_to_save=["classifier"],
        bias="none",
    )

    return get_peft_model(base_model, peft_config)


def count_trainable_parameters(model) -> dict[str, int | float]:
    """Return a breakdown of total, trainable, and frozen parameter counts.

    Works with any nn.Module, including PEFT-wrapped models.

    Returns:
        dict with keys:
            "total"           — total number of parameters.
            "trainable"       — number of parameters with requires_grad=True.
            "frozen"          — number of frozen parameters.
            "trainable_ratio" — trainable / total (rounded to 6 decimal places).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": round(trainable / total, 6) if total > 0 else 0.0,
    }
